"""Generate synthetic structure factors from a multi-model ensemble, one model at a time.

The PDBFixer-based synthetic data pipeline writes each altloc conformer as a separate model
of one multi-model file, every model a complete copy of the protein. The other script
`generate_synthetic_sf.py`` computes a structure's structure factor in a single pass from a
single-model structure with altlocs. The total number of atoms in that case is N_blank_altloc
+ N_conformer * N_non_blank_altloc. In contrast, the multi-model file consists of N_conformer
*(N_blank_altloc + N_non_blank_altloc) atoms. This can easily exhaust memory usage as SFC
holds an ``[N_atom, N_HKL]`` tensor of per-atom plus several intermediates of that shape.

This script instead computes Fprotein from one model at a time and sums the results
weighted by each model's population, so only one model's ``[N_atom, N_HKL]`` tensors are
alive at once. Every model must have the same atoms in the same order.

Bulk solvent is optional (``--bulk-solvent``), with the same modes as the structure-factor
loss. The bulk solvent masks are built from Fprotein rather than directly from atoms:
``combined`` builds one bulk solvent mask from the population-weighted Fprotein sum, and
``per_conformer`` builds one bulk solvent mask per model and takes the population-weighted
mean of the bulk solvent structure factors. Either mode writes Ftotal/SIGFtotal/PHIFtotal
columns (at default unrefined SFcalculator scales) beside the Fprotein columns in the MTZ.

The command-line interface mirrors ``generate_synthetic_sf.py``, including CSV batch mode,
with ``--occupancy-values`` (or the ``occupancy_values`` CSV column) giving one population
per model.

The processed ensemble (after filtering and B-factor override) can optionally be saved as
``<input_stem>_sf_input.cif`` beside the MTZ, one model per conformer at occupancy 1.0 so
each model is a standalone structure; the populations are recorded only in the occupancy
values.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import cast

import gemmi
import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.utils
import torch
from loguru import logger
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser

from sampleworks.synthetic.generate_synthetic_sf import (
    BatchRowForMTZ,
    load_batch_csv,
)
from sampleworks.synthetic.synthetic_utils import (
    atomarray_to_gemmi,
    load_structure_for_synthetic_loss,
    resolve_crystal_metadata,
    resolve_mtz_column,
    resolve_parallel_jobs,
)
from sampleworks.utils.torch_utils import try_gpu


# Same modes as the structure-factor loss's ``bulk_solvent`` option.
BULK_SOLVENT_MODES = ("off", "combined", "per_conformer")
OCCUPANCY_MODES = ("default", "uniform", "custom")


def write_sf_input_structure(
    gemmi_structure: gemmi.Structure, structure_path: Path, output_dir: Path
) -> Path:
    """Save the structure that structure factors were computed from.

    Parameters
    ----------
    gemmi_structure
        Processed structure passed to SFcalculator.
    structure_path
        Source structure path whose stem names the output.
    output_dir
        Directory in which to write ``<source_stem>_sf_input.cif``.

    Returns
    -------
    Path
        Written mmCIF path.

    Notes
    -----
    This function creates ``output_dir`` and writes a file.
    """
    output_path = output_dir / f"{structure_path.stem}_sf_input.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gemmi_structure.make_mmcif_document().write_file(str(output_path))
    logger.info(f"Saved processed structure to {output_path}")
    return output_path


def _build_rs_dataset_for_one_label(
    sfc: SFcalculator,
    label: str,
    structure_factor_column: str,
    miller_index_column: str,
    sigma_f_scale: float,
) -> rs.DataSet:
    """Build an rs.DataSet that contains only one set of F / SIGF / PHIF columns.

    ``sfc.prepare_dataset`` returns an amplitude column and a phase column (degrees)
    for the given ``structure_factor_column`` attribute. We auto-detect those by MTZ
    dtype (rather than assuming the unexposed ``FMODEL`` / ``PHIFMODEL`` names),
    rename them to ``F{label}`` / ``PHIF{label}``, and synthesize a ``SIGF{label}``
    column so several structure-factor sets (e.g. protein and total) can coexist in
    one MTZ.

    Parameters
    ----------
    sfc : SFcalculator
        Structure-factor calculator holding the requested ASU amplitudes and phases.
    label : str
        Mtz column suffix. The resulted column names are ``F{label}``/``SIGF{label}``/
        ``PHIF{label}``. The mtz column names should match what the structure-factor reward
        later reads. In this script, we use ``"protein"`` and ``"total"`` as the labels.
    structure_factor_column : str
        SFcalculator attribute name passed to ``prepare_dataset`` (the amplitudes to emit).
    miller_index_column : str
        SFcalculator attribute name holding the Miller indices for ``prepare_dataset``.
    sigma_f_scale : float
        Multiplier used to synthesize the ``SIGF{label}`` column from the amplitudes.

    Returns
    -------
    rs.DataSet
        Dataset with columns ``F{label}``, ``SIGF{label}``, ``PHIF{label}`` (in that order).
    """
    dataset: rs.DataSet = sfc.prepare_dataset(miller_index_column, structure_factor_column)
    amplitude_column = resolve_mtz_column(dataset, rs.StructureFactorAmplitudeDtype())
    phase_column = resolve_mtz_column(dataset, rs.PhaseDtype())
    logger.debug(
        f"Auto-detected amplitude column: {amplitude_column}, "
        f"phase column: {phase_column} for {label}"
    )
    f_col, phi_col, sig_col = f"F{label}", f"PHIF{label}", f"SIGF{label}"
    # rs.DataSet.rename returns a DataSet at runtime, but the stub types it as DataFrame.
    dataset = cast(
        rs.DataSet,
        dataset.rename(columns={amplitude_column: f_col, phase_column: phi_col}),
    )
    dataset[sig_col] = (dataset[f_col] * sigma_f_scale).astype(rs.StandardDeviationDtype())
    return dataset[[f_col, sig_col, phi_col]]


def process_amplitudes_to_dataset(
    sfc: SFcalculator,
    structure_factor_columns: dict[str, str],
    test_fraction: float = 0.05,
    seed: int | None = None,
    miller_index_column: str = "Hasu_array",
    ccp4_convention: bool = False,
    sigma_f_scale: float = 0.2,
    output_path: Path | None = None,
) -> rs.DataSet:
    """Build an rs.DataSet from SFcalculator outputs, optionally writing it as MTZ.

    Parameters
    ----------
    sfc: SFcalculator
        SFcalculator instance
    structure_factor_columns: dict[str, str]
        Mapping of ``label -> SFcalculator attribute``. Label must match the pattern of
        SFcalculator attribute ``F{label}_asu`` (e.g. ``protein`` ->  ``Fprotein_asu``).
        One structure-factor set (``F{label}``/``SIGF{label}``/``PHIF{label}``) is
        emitted per label, and multiple labels are merged into one MTZ sharing the same
        HKL list (``miller_index_column``). For example, ``{"protein": "Fprotein_asu",
        "total": "Ftotal_asu"}`` produces ``Fprotein``/``SIGFprotein``/``PHIFprotein``/
        ``Ftotal``/``SIGFtotal``/``PHIFtotal`` and optionallyan R-free flag column.
    test_fraction: float
        Fraction of reflections to mark as R-free test set (0 disables)
    seed: int | None
        Optional seed for reproducible R-free flag assignment
    miller_index_column: str
        Attribute name in SFcalculator for hkl indices
    ccp4_convention: bool
        If True, use CCP4 convention for R-free flag assignment. Default
        is False, which uses Phenix convention (1 = test, 0 = working).
    sigma_f_scale: float
        Scale factor to make a fake sigma column from the structure factor amplitude
        values so that SFcalculator can load the output MTZ file as synthetic reward
        without errors. The actual sigma values only matter when computing R-factor.
    output_path: Path | None
        If provided, write the dataset to this MTZ file path.

    Returns
    -------
    rs.DataSet
        Dataset with structure factor amplitudes, dummy sigma column(s), phases,
        and optionally R-free flags.
    """
    # One dataset per label. Labels are free-form column suffixes (F{label}/SIGF{label}/
    # PHIF{label}). User should supply the resulted column names when using the mtz for
    # structure-factor reward. In this script, we use "protein" and "total" (for protein
    # and bulk solvent) as the labels.
    # All sets share the same HKL index, so this is just a column-wise concat.
    dataset: rs.DataSet | None = None
    for label, attribute in structure_factor_columns.items():
        ds = _build_rs_dataset_for_one_label(
            sfc, label, attribute, miller_index_column, sigma_f_scale
        )
        if dataset is None:
            dataset = ds
        else:
            collisions = dataset.columns.intersection(ds.columns).tolist()
            if collisions:
                raise ValueError(
                    f"Structure-factor label {label!r} would overwrite already-populated "
                    f"column(s) {collisions} in the merged dataset. Ensure each label maps "
                    "to a distinct set of column names."
                )
            dataset[ds.columns] = ds  # graft the remaining columns on (index-aligned)
    if dataset is None:  # loop never ran -> the mapping was empty
        raise ValueError(
            "structure_factor_columns must contain at least one "
            "'label -> SFcalculator attribute' entry (e.g. {'protein': 'Fprotein_asu'}), "
            f"got {structure_factor_columns!r}."
        )
    if test_fraction > 0:
        dataset = rs.utils.add_rfree(
            dataset,
            ccp4_convention=ccp4_convention,
            fraction=test_fraction,
            seed=seed,
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_mtz(str(output_path))
        logger.info(f"Saved structure factors to {output_path}")
    return dataset


def compute_ensemble_structure_factors(
    structure_path: Path,
    occupancies: list[float],
    cell: gemmi.UnitCell,
    space_group: str,
    resolution: float,
    scattering_factor_mode: str,
    device: torch.device,
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    selection: str | None = None,
    b_factor: float | None = None,
    save_structure_dir: Path | None = None,
    bulk_solvent: str = "off",
) -> SFcalculator:
    """Sum each model's Fprotein onto one reflection list, optionally adding bulk solvent.

    Parameters
    ----------
    structure_path
        Multi-model structure file holding one conformer per model.
    occupancies
        Population per model, in model order, written over every atom's occupancy.
    cell
        Unit cell shared by the ensemble.
    space_group
        Hermann-Mauguin space group shared by the ensemble.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem".
    device
        Torch device for the calculation.
    strip_hydrogens
        If True, remove hydrogen atoms from every model.
    strip_waters
        If True, remove water molecules from every model.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    selection
        Optional atom selection applied to every model.
    b_factor
        Isotropic B-factor written over every atom, or None to keep the file's values.
    save_structure_dir
        Optional directory in which to save the processed ensemble as mmCIF, one model
        per conformer at occupancy 1.0 so each model is a standalone structure.
    bulk_solvent
        ``"off"`` (Fprotein only), ``"combined"`` (one mask from the occupancy-weighted
        Fprotein sum), or ``"per_conformer"`` (occupancy-weighted mean of per-model
        masks).

    Returns
    -------
    SFcalculator
        The last model's sfcalculator, carrying the ensemble sum in ``Fprotein_asu``,
        the solvent in ``Fmask_asu`` (zero when ``bulk_solvent="off"``), and, unless
        ``"off"``, ``Ftotal_asu`` at default scales. Ready for
        ``process_amplitudes_to_dataset``.

    Raises
    ------
    ValueError
        If ``occupancies`` is empty, a model cannot be loaded, the models do not land on
        the same reflection list, or ``bulk_solvent`` is not a known mode.

    Notes
    -----
    Only one model's ``[N_atom, N_HKL]`` per-atom structure-factor tensors are alive at a
    time; each model's SFcalculator is released before the next is built. Each model's
    Fprotein is computed at occupancy 1.0 and weighted by its population afterwards.

    The solvent fraction is estimated from model 1's atom positions, as the
    structure-factor loss estimates it from the guidance reference, which is model 1
    of a multi-model input. ``per_conformer`` goes through
    ``SFcalculator.calc_fsolvent_batch``, the path the structure-factor loss uses, so
    it inherits that path's batch behavior: models are processed in chunks of
    ``calc_fsolvent_batch``'s default ``PARTITION=10`` (also the loss's default
    ``batch_partition``), the density is normalized over each whole chunk, and the
    solvent cutoff comes from the chunk's first model. Unlike the loss, which
    weights every conformer's mask by ``1/batch_size``, the masks here are weighted by
    occupancy; the two agree for uniform occupancies.
    """
    if bulk_solvent not in BULK_SOLVENT_MODES:
        raise ValueError(f"bulk_solvent must be one of {BULK_SOLVENT_MODES}, got {bulk_solvent!r}")
    if not occupancies:
        raise ValueError(f"Need at least one occupancy (one per model) for {structure_path}")
    model_fproteins = []  # one [n_hkl] complex Fprotein_asu per model, at occupancy 1.0
    reference_hkl = None  # [n_hkl, 3] ASU Miller indices every model must land on
    saved_structure = None
    reference_solventpct = None  # solvent fraction of model 1, estimated from its atoms
    for model, occupancy in enumerate(occupancies):
        sfcalculator = None  # release the previous model's scattering table first
        torch.cuda.empty_cache()
        atom_array = load_structure_for_synthetic_loss(
            structure_path,
            occupancy_mode="default",
            occupancy_values=[],
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            selection=selection,
            b_factor=b_factor,
            model=model,
        )
        if atom_array is None:
            raise ValueError(f"Failed to load model {model + 1} of {structure_path}")
        # Each model stands alone at full occupancy; its population lives in ``occupancies``.
        atom_array.occupancy[:] = 1.0
        gemmi_structure = atomarray_to_gemmi(atom_array, cell, space_group)
        if save_structure_dir is not None:
            if saved_structure is None:
                saved_structure = gemmi_structure.clone()
            else:
                saved_structure.add_model(gemmi_structure[0])  # copies the model
                saved_structure[len(saved_structure) - 1].name = str(model + 1)
        logger.info(f"Model {model + 1}: {len(atom_array)} atoms at population {occupancy:g}")
        sfcalculator = SFcalculator(
            pdbmodel=PDBParser(gemmi_structure),
            mtzdata=None,
            dmin=resolution,
            mode=scattering_factor_mode,
            anomalous=False,
            set_experiment=False,
            device=device,
        )
        if reference_hkl is None:
            reference_hkl = sfcalculator.Hasu_array
        elif not np.array_equal(reference_hkl, sfcalculator.Hasu_array):
            raise ValueError(
                f"Model {model + 1} of {structure_path} produced a different reflection "
                "list than model 1; the models cannot be summed."
            )
        if model == 0 and bulk_solvent != "off":
            sfcalculator.inspect_data()
            reference_solventpct = sfcalculator.solventpct
        sfcalculator.calc_fprotein()
        model_fproteins.append(sfcalculator.Fprotein_asu)

    if save_structure_dir is not None and saved_structure is not None:
        write_sf_input_structure(saved_structure, structure_path, save_structure_dir)
    populations = torch.tensor(occupancies, device=sfcalculator.device)[:, None]  # [n_models, 1]
    fprotein_batch = torch.stack(model_fproteins)  # [n_models, n_hkl] complex
    sfcalculator.Fprotein_asu = (populations * fprotein_batch).sum(dim=0)
    if bulk_solvent == "off":
        sfcalculator.Fmask_asu = torch.zeros_like(sfcalculator.Fprotein_asu)
        return sfcalculator

    sfcalculator.inspect_data()  # grid size, from the shared reflection list
    sfcalculator.solventpct = reference_solventpct
    if bulk_solvent == "combined":
        sfcalculator.calc_fsolvent()  # sets Fmask_asu from the summed Fprotein_asu
    else:
        sfcalculator.Fprotein_asu_batch = fprotein_batch
        fmask_batch = sfcalculator.calc_fsolvent_batch(Return=True)  # [n_models, n_hkl] complex
        sfcalculator.Fmask_asu = (populations * fmask_batch).sum(dim=0)
    sfcalculator.init_scales(requires_grad=False)
    sfcalculator.calc_ftotal()  # sets Ftotal_asu
    return sfcalculator


def _process_single_row(
    row: BatchRowForMTZ,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    occupancy_mode: str = "default",
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
    bulk_solvent: str = "off",
) -> None:
    """Compute and write the structure factors of one structure file. Assume no anomalous
    scattering.

    Failures are logged rather than raised so one bad row does not stop a batch.

    Parameters
    ----------
    row
        BatchRowForMTZ naming the structure file, with optional occupancy values (per
        altloc for single-model, or per model otherwise) and per-row overrides.
    base_dir
        Base directory for resolving relative structure file paths.
    output_dir
        Directory where the MTZ (and saved structure) will be written.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        SFcalculator mode: "xray" or "cryoem".
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables).
    seed
        Optional seed for reproducible R-free flag assignment.
    device
        PyTorch device for SFcalculator.
    occupancy_mode
        'default', 'uniform', or 'custom'; see ``compute_ensemble_structure_factors``.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save the processed structure as ``<input_stem>_sf_input.cif`` beside
        the MTZ.
    b_factor
        Optional isotropic B-factor assigned to every retained atom.
    bulk_solvent
        Bulk-solvent mode: ``"off"``, ``"combined"``, or ``"per_conformer"``; see
        ``compute_ensemble_structure_factors``.

    Notes
    -----
    Writes the MTZ ``row.mtzfile`` (default ``<input_stem>_<res>A.mtz``) to output_dir.
    The columns are ``Fprotein``/``SIGFprotein``/``PHIFprotein`` plus optional R-free
    flags, and, unless ``bulk_solvent="off"``, ``Ftotal``/``SIGFtotal``/``PHIFtotal``.
    """
    structure_path = base_dir / row.filename
    output_path = output_dir / (row.mtzfile or f"{structure_path.stem}_{resolution:.2f}A.mtz")
    try:
        gemmi_meta = gemmi.read_structure(str(structure_path))
        if len(row.occupancy_values) != len(gemmi_meta):
            raise ValueError(
                f"Expected one occupancy per model, got {len(row.occupancy_values)} for "
                f"{len(gemmi_meta)} models."
            )
        cell, space_group = resolve_crystal_metadata(row, structure_path)
        logger.info(f"Cell {cell.parameters}, space group '{space_group}'")
        sfcalculator = compute_ensemble_structure_factors(
            structure_path,
            row.occupancy_values,
            cell,
            space_group,
            resolution,
            scattering_factor_mode,
            device,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            selection=row.selection,
            b_factor=b_factor,
            save_structure_dir=output_path.parent if save_structure else None,
            bulk_solvent=bulk_solvent,
        )
    except Exception as e:
        logger.error(
            f"Failed to compute for {row.filename} ({type(e).__name__}): {e}\n"
            f"{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return

    structure_factor_columns = {"protein": "Fprotein_asu"}
    if bulk_solvent != "off":
        structure_factor_columns["total"] = "Ftotal_asu"
    try:
        process_amplitudes_to_dataset(
            sfcalculator,
            structure_factor_columns=structure_factor_columns,
            test_fraction=test_fraction,
            seed=seed,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(
            f"Failed to write MTZ for {row.filename} to {output_path} ({type(e).__name__}): "
            f"{e}\n{''.join(traceback.format_tb(e.__traceback__))}"
        )
        return
    logger.info(f"Wrote {row.filename} (bulk solvent {bulk_solvent}) to {output_path}")


def process_batch(
    csv_path: Path,
    base_dir: Path,
    output_dir: Path,
    resolution: float,
    scattering_factor_mode: str,
    test_fraction: float,
    seed: int | None,
    device: torch.device,
    n_jobs: int = -1,
    occupancy_mode: str = "default",
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
    bulk_solvent: str = "off",
) -> None:
    """Process multiple structure files from a CSV file in batch mode.

    Parameters
    ----------
    csv_path
        Path to CSV file listing structures to process, one structure file per row.
    base_dir
        Base directory for resolving relative structure file paths.
    output_dir
        Directory where output MTZ files will be written.
    resolution
        High-resolution (dmin) limit in Angstroms.
    scattering_factor_mode
        Scattering factor type: 'xray' or 'cryoem'.
    test_fraction
        Fraction of reflections to mark as R-free test set (0 disables).
    seed
        Optional seed for reproducible R-free flag assignment.
    device
        PyTorch device for computation.
    n_jobs
        Number of parallel jobs. -1 means use all available CPUs.
    occupancy_mode
        'default', 'uniform', or 'custom'; see ``compute_ensemble_structure_factors``.
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save each processed structure as mmCIF beside its MTZ.
    b_factor
        Optional isotropic B-factor assigned to every retained atom.
    bulk_solvent
        Bulk-solvent mode: ``"off"``, ``"combined"``, or ``"per_conformer"``.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} structures from {csv_path} using {effective_n_jobs} jobs")

    Parallel(n_jobs=effective_n_jobs, backend="loky")(
        delayed(_process_single_row)(
            row=row,
            base_dir=base_dir,
            output_dir=output_dir,
            resolution=resolution,
            scattering_factor_mode=scattering_factor_mode,
            test_fraction=test_fraction,
            seed=seed,
            device=device,
            occupancy_mode=occupancy_mode,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            save_structure=save_structure,
            b_factor=b_factor,
            bulk_solvent=bulk_solvent,
        )
        for row in rows
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed structure factor configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sum Fprotein (and optionally bulk solvent) over the models of a multi-model "
            "ensemble file and write one synthetic MTZ, without building a merged altloc "
            "structure"
        )
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--structure",
        "-s",
        type=Path,
        help="Multi-model structure file (mmCIF or PDB) with one conformer per model",
    )
    input_group.add_argument("--batch-csv", type=Path, help="Path to CSV file for batch processing")
    input_group.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths in CSV, not used in single-structure mode",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection applied to every model (e.g., 'chain A and resi 10-50')",
    )

    occupancy_group = parser.add_argument_group("Occupancy Options")
    occupancy_group.add_argument(
        "--occupancy-mode",
        choices=OCCUPANCY_MODES,
        default="default",
        help=(
            "Occupancy assignment mode. One model: altloc occupancies (default keeps the "
            "file's, uniform gives 1/n_altlocs). Multi-model: model populations (uniform "
            "gives 1/n_models; default is refused)"
        ),
    )
    occupancy_group.add_argument(
        "--occupancy-values",
        type=str,
        help=(
            "Colon-separated values summing to 1 (e.g., '0.3:0.7'): altloc occupancies in "
            "sorted altloc order for one model, or one occupancy per model"
        ),
    )

    sf_group = parser.add_argument_group("Structure Factor Options")
    sf_group.add_argument(
        "--resolution",
        "-r",
        type=float,
        default=1.0,
        help="High-resolution (dmin) limit in Angstroms",
    )
    sf_group.add_argument(
        "--b-factor",
        type=float,
        default=None,
        help="Override every retained atom's isotropic B-factor",
    )
    sf_group.add_argument(
        "--scattering-factor-mode",
        choices=["xray", "cryoem"],
        default="xray",
        help="Scattering factor type",
    )
    sf_group.add_argument(
        "--remove-hydrogens",
        action="store_true",
        help="Remove hydrogen atoms before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-waters",
        action="store_true",
        help="Remove water molecules before computing structure factors",
    )
    sf_group.add_argument(
        "--remove-ligands",
        action="store_true",
        help="Remove ligand molecules (non-water heteroatoms) before computing structure factors",
    )
    sf_group.add_argument(
        "--bulk-solvent",
        choices=BULK_SOLVENT_MODES,
        default="off",
        help=(
            "Bulk-solvent treatment: off (Fprotein only), combined (one mask from the "
            "population-weighted ensemble density), or per_conformer (population-weighted "
            "mean of per-model masks). Either solvent mode also writes Ftotal at default scales"
        ),
    )

    rfree_group = parser.add_argument_group("R-free Options")
    rfree_group.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Fraction of reflections flagged as R-free test set (0 disables)",
    )
    rfree_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible R-free flag assignment",
    )

    crystal_group = parser.add_argument_group("Crystal Options (single-structure mode only)")
    crystal_group.add_argument(
        "--unit-cell",
        type=str,
        help="Unit cell as 'a:b:c:alpha:beta:gamma' (overrides the file's CRYST1 record)",
    )
    crystal_group.add_argument(
        "--space-group",
        type=str,
        help="Space group as Hermann-Mauguin string or number (overrides CRYST1 record)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-structure",
        action="store_true",
        help="Save the processed structure as <input_stem>_sf_input.cif beside the MTZ",
    )
    output_group.add_argument("--output", "-o", type=Path, help="Output MTZ file path")
    output_group.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Output directory for batch mode"
    )

    parallel_group = parser.add_argument_group("Parallelization Options")
    parallel_group.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for batch processing (-1 uses all CPUs)",
    )

    return parser.parse_args()


def main() -> None:
    """Generate synthetic MTZs using command-line arguments."""
    args = parse_args()
    device = try_gpu()

    if args.batch_csv:
        process_batch(
            csv_path=args.batch_csv,
            base_dir=args.base_dir,
            output_dir=args.output_dir,
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            n_jobs=args.n_jobs,
            occupancy_mode=args.occupancy_mode,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
            b_factor=args.b_factor,
            bulk_solvent=args.bulk_solvent,
        )
    elif args.structure:
        row = BatchRowForMTZ.from_dict(
            {
                "filename": args.structure.name,
                "mtzfile": args.output.name if args.output else None,
                "unit_cell": args.unit_cell,
                "space_group": args.space_group,
                "selection": args.selection,
                "occupancy_values": args.occupancy_values,
            }
        )
        _process_single_row(
            row=row,
            base_dir=args.structure.parent,
            output_dir=args.output.parent if args.output else Path("."),
            resolution=args.resolution,
            scattering_factor_mode=args.scattering_factor_mode,
            test_fraction=args.test_fraction,
            seed=args.seed,
            device=device,
            occupancy_mode=args.occupancy_mode,
            strip_hydrogens=args.remove_hydrogens,
            strip_waters=args.remove_waters,
            strip_ligands=args.remove_ligands,
            save_structure=args.save_structure,
            b_factor=args.b_factor,
            bulk_solvent=args.bulk_solvent,
        )
    else:
        logger.error("Please specify --structure or --batch-csv")
        sys.exit(1)


if __name__ == "__main__":
    main()
