"""Generate synthetic structure factor amplitudes via SFcalculator-torch.

Accepts two kinds of input, told apart by the number of models in the file:

- **Multi-conformer** (one model with at least 2 altloc IDs): the altloc conformers share
  one model, so the structure factors are computed in a single pass. ``--occupancy-mode``
  and ``--occupancy-values`` set the altloc occupancies.
- **Multi-model** (the PDBFixer-based preprocessing pipeline): each conformer is a
  complete copy of the protein in its own model. ``--occupancy-values`` represents population
  of the conformation (i.e. occupancy of a model rather than of an atom).
  ``--occupancy-mode default`` is ill-posed and refused in this case.

A multi-model file holds N_conformer * (N_blank_altloc + N_non_blank_altloc) atoms, against
N_blank_altloc + N_conformer * N_non_blank_altloc for the equivalent multi-conformer model.
SFC holds an ``[N_atom, N_HKL]`` tensor of per-atom structure factors plus several
intermediates of that shape, so merging the models would easily exhaust memory. Fprotein
is instead computed from one model at a time and summed weighted by population (each model's
occupancy), so only one model's ``[N_atom, N_HKL]`` tensors are alive at once.

Bulk solvent is optional (``--bulk-solvent``), with the same modes as the structure-factor
loss. The bulk solvent masks are built from Fprotein rather than directly from atoms:
``combined`` builds one bulk solvent mask from the population-weighted Fprotein sum, and
``per_conformer`` builds one bulk solvent mask per model and takes the population-weighted
mean of the bulk solvent structure factors. Either mode writes Ftotal/SIGFtotal/PHIFtotal
columns (at default unrefined SFcalculator scales) beside the Fprotein/SIGFprotein/
PHIFprotein columns in the MTZ. The SIGF columns hold dummy values, and an R-free flag
column is optionally added. For a single model the two solvent modes are equivalent.

Each structure can be optionally overridden with unit cell, space group, atom selection,
and a flat isotropic B-factor, and the processed structure can be saved as
``<input_stem>_sf_input.cif`` beside the MTZ. A saved multi-model file keeps one model
per conformer with every atom at occupancy 1.0, so each model is a standalone structure.
The per-model occupancies (conformer population) are not written into it, because biotite
holds one occupancy annotation shared by all models in an ``AtomArrayStack``.
"""

import argparse
import sys
import traceback
from functools import partial
from pathlib import Path
from typing import cast

import gemmi
import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.utils
import torch
from biotite.structure import AtomArray
from loguru import logger
from SFC_Torch import SFcalculator
from SFC_Torch.io import PDBParser

from sampleworks.synthetic.synthetic_utils import (
    atomarray_to_gemmi,
    BatchRowForMTZ,
    resolve_occupancy_values,
    load_batch_csv,
    load_structure_for_synthetic_loss,
    resolve_crystal_metadata,
    resolve_mtz_column,
    resolve_parallel_jobs,
)
from sampleworks.utils.atom_array_utils import detect_altlocs
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
        ``PHIF{label}``. The mtz column names should match what the structure-factor loss
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
        values so that SFcalculator can load the output MTZ file as synthetic loss
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
    # structure-factor loss. In this script, we use "protein" and "total" (for protein
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


def load_models(
    structure_path: Path,
    occupancy_mode: str,
    occupancy_values: list[float],
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    selection: str | None = None,
    b_factor: float | None = None,
) -> tuple[list[AtomArray], list[float]]:
    """Load a structure file as models and the population that weights each one.

    Parameters
    ----------
    structure_path
        Structure file: one multi-conformer model, or one conformer per model.
    occupancy_mode
        'default', 'uniform', or 'custom'. Sets the altloc occupancies of a single model,
        or the population of each model of a multi-model file, where 'default' is refused.
    occupancy_values
        Altloc occupancies (single model) or one population per model (multi-model).
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

    Returns
    -------
    tuple[list[AtomArray], list[float]]
        One atom array per model, and the population weighting each model: ``[1.0]`` for
        a single model, whose altloc occupancies are on its atoms. Every atom of a
        multi-model file is at occupancy 1.0.

    Raises
    ------
    ValueError
        If the file has no models or a model cannot be loaded, or, for a multi-model file,
        the populations cannot be resolved or a model uses more than one non-blank altloc ID.
    """
    n_models = len(gemmi.read_structure(str(structure_path)))
    load_filtered = partial(
        load_structure_for_synthetic_loss,
        strip_hydrogens=strip_hydrogens,
        strip_waters=strip_waters,
        strip_ligands=strip_ligands,
        selection=selection,
        b_factor=b_factor,
    )
    if n_models == 0:
        raise ValueError(f"{structure_path} holds no models.")
    if n_models == 1:
        atom_array = load_filtered(structure_path, occupancy_mode, occupancy_values)
        if atom_array is None:
            raise ValueError(f"Failed to load {structure_path}")
        return [atom_array], [1.0]

    if occupancy_values and occupancy_mode != "custom":
        logger.warning(
            f"Custom occupancy values provided for {structure_path}, "
            f"but occupancy_mode is '{occupancy_mode}'. Using 'custom' mode."
        )
        occupancy_mode = "custom"
    if occupancy_mode == "uniform":
        populations = [1.0 / n_models] * n_models
    elif occupancy_mode == "custom":
        if not occupancy_values:
            raise ValueError(
                f"occupancy_values required for custom mode: pass one population per model "
                f"of the {n_models}-model file."
            )
        populations = resolve_occupancy_values(occupancy_values, n_models, "models")
    elif occupancy_mode == "default":
        raise ValueError(
            f"Occupancy mode 'default' is not supported for a {n_models}-model file; pass one "
            "population per model or use 'uniform'."
        )
    else:
        raise ValueError(f"Invalid occupancy mode '{occupancy_mode}'")

    models = []
    for model in range(n_models):
        # Keep the file's per-atom occupancies; each model is set to 1.0 below.
        atom_array = load_filtered(structure_path, "default", [], model=model)
        if atom_array is None:
            raise ValueError(f"Failed to load model {model + 1} of {structure_path}")
        altloc_ids = detect_altlocs(atom_array).altloc_ids
        if len(altloc_ids) > 1:
            raise ValueError(
                f"Model {model + 1} of {structure_path} has altloc IDs {altloc_ids}. A "
                "multi-model structure is expected to have at most one non-blank altloc ID "
                "per model for evaluation scripts; it should not have multiple altloc IDs "
                "that represent distinct conformations."
            )
        atom_array.occupancy[:] = 1.0
        models.append(atom_array)
    return models, populations


def compute_ensemble_structure_factors(
    models: list[AtomArray],
    populations: list[float],
    cell: gemmi.UnitCell,
    space_group: str,
    resolution: float,
    scattering_factor_mode: str,
    device: torch.device,
    bulk_solvent: str = "off",
) -> SFcalculator:
    """Sum each model's Fprotein onto one reflection list, optionally adding bulk solvent.

    Parameters
    ----------
    models
        One atom array per model, as returned by ``load_models``.
    populations
        Population weighting each model, in the same order.
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
    bulk_solvent
        ``"off"`` (Fprotein only), ``"combined"`` (one mask from the population-weighted
        Fprotein sum), or ``"per_conformer"`` (population-weighted mean of per-model
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
        If the models do not land on the same reflection list, or ``bulk_solvent``
        is not a known mode.

    Notes
    -----
    Only one model's ``[N_atom, N_HKL]`` per-atom structure-factor tensors are alive
    at a time; each model's SFcalculator is released before the next is built.

    The solvent fraction is estimated from model 1's atom positions, as the
    structure-factor loss estimates it from the guidance reference, which is model 1
    of a multi-model input. ``per_conformer`` goes through
    ``SFcalculator.calc_fsolvent_batch``, the path the structure-factor loss uses, so
    it inherits that path's batch behavior: models are processed in chunks of
    ``calc_fsolvent_batch``'s default ``PARTITION=10`` (also the loss's default
    ``batch_partition``), the density is normalized over each whole chunk, and the
    solvent cutoff comes from the chunk's first model. Unlike the loss, which
    weights every model's mask by ``1/batch_size``, the masks here are weighted by
    population; the two agree for uniform populations.
    """
    if bulk_solvent not in BULK_SOLVENT_MODES:
        raise ValueError(f"bulk_solvent must be one of {BULK_SOLVENT_MODES}, got {bulk_solvent!r}")
    model_fproteins = []  # one [n_hkl] complex Fprotein_asu per model
    reference_hkl = None  # [n_hkl, 3] ASU Miller indices every model must land on
    reference_solventpct = None  # solvent fraction of model 1, estimated from its atoms
    for model, (atom_array, population) in enumerate(zip(models, populations, strict=True)):
        sfcalculator = None  # release the previous model's scattering table first
        torch.cuda.empty_cache()
        logger.info(f"Model {model + 1}: {len(atom_array)} atoms at population {population:g}")
        sfcalculator = SFcalculator(
            pdbmodel=PDBParser(atomarray_to_gemmi(atom_array, cell, space_group)),
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
                f"Model {model + 1} produced a different reflection list than model 1; "
                "the models cannot be summed."
            )
        if model == 0 and bulk_solvent != "off":
            sfcalculator.inspect_data()
            reference_solventpct = sfcalculator.solventpct
        sfcalculator.calc_fprotein()
        model_fproteins.append(sfcalculator.Fprotein_asu)

    # [n_models, 1]
    population_weights = torch.tensor(populations, device=sfcalculator.device)[:, None]
    fprotein_batch = torch.stack(model_fproteins)  # [n_models, n_hkl] complex
    sfcalculator.Fprotein_asu = (population_weights * fprotein_batch).sum(dim=0)
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
        sfcalculator.Fmask_asu = (population_weights * fmask_batch).sum(dim=0)
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
        cell, space_group = resolve_crystal_metadata(row, structure_path)
        logger.info(f"Cell {cell.parameters}, space group '{space_group}'")
        models, populations = load_models(
            structure_path,
            occupancy_mode,
            row.occupancy_values,
            strip_hydrogens=strip_hydrogens,
            strip_waters=strip_waters,
            strip_ligands=strip_ligands,
            selection=row.selection,
            b_factor=b_factor,
        )
        if save_structure:
            saved_structure = atomarray_to_gemmi(models[0], cell, space_group)
            for model, atom_array in enumerate(models[1:], start=2):
                saved_structure.add_model(atomarray_to_gemmi(atom_array, cell, space_group)[0])
                saved_structure[len(saved_structure) - 1].name = str(model)
            write_sf_input_structure(saved_structure, structure_path, output_path.parent)
        sfcalculator = compute_ensemble_structure_factors(
            models,
            populations,
            cell,
            space_group,
            resolution,
            scattering_factor_mode,
            device,
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
            "Generate synthetic structure factor amplitudes from a multi-conformer (altloc) "
            "structure or a multi-model ensemble"
        )
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--structure",
        "-s",
        type=Path,
        help="Structure file (mmCIF or PDB): multi-conformer structure or multi-model structure",
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
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for outputs, in both single-structure and batch mode",
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
            output_dir=args.output.parent if args.output else args.output_dir,
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
