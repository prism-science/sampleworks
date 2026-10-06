"""Generate synthetic structure factors from a multi-model ensemble, one model at a time.

``generate_synthetic_sf.py`` computes one structure's structure factors in a single pass,
which means one ``[N_atom, N_HKL]`` scattering table and a handful of intermediates of the
same shape. The atoms are mostly blank-altloc atoms plus a few atoms in altlocs. In
the latest synthetic data generation pipeline that uses PDBFixer, each altloc ends up as a
separate conformer, and the conformers are collected as models of one multi-model file.
Even though they can be trivially merged back as one structure where all atoms have
altlocs, such a structure easily leads to OOM.

This script computes Fprotein from one model at a time to reduce memory usage, and then
sums Fprotein weighted by each model's population. Every model must have the same atoms in
the same order.

Bulk solvent is optional (``--bulk-solvent``), with the same modes as the structure-factor
reward. Both masks are built from Fprotein rather than from atoms, so neither needs a merged
structure: ``combined`` masks the population-weighted Fprotein sum (one mask from the
ensemble density), and ``per_conformer`` masks each model's Fprotein separately and takes
the population-weighted mean of those masks' structure factors. Either mode adds an Ftotal
set (default, unrefined SFcalculator scales) beside Fprotein in the MTZ.

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

import gemmi
import numpy as np
import torch
from loguru import logger
from SFC_Torch import SFcalculator

from sampleworks.synthetic.generate_synthetic_sf import (
    BatchRowForMTZ,
    build_sfcalculator,
    load_batch_csv,
    process_amplitudes_to_dataset,
    write_sf_input_structure,
)
from sampleworks.synthetic.synthetic_utils import (
    atomarray_to_gemmi,
    load_structure_for_synthetic_reward,
    resolve_parallel_jobs,
)
from sampleworks.utils.torch_utils import try_gpu


# Same modes as the structure-factor reward's ``bulk_solvent`` option.
BULK_SOLVENT_MODES = ("off", "combined", "per_conformer")


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
        If a model cannot be loaded, the models do not land on the same reflection list,
        or ``bulk_solvent`` is not a known mode.

    Notes
    -----
    Only one model's scattering table is alive at a time; each is released before the
    next is built. Each model's Fprotein is computed at occupancy 1.0 and weighted by
    its population afterwards.

    The solvent fraction is estimated from model 1's atom positions, as the
    structure-factor reward estimates it from the guidance reference, which is model 1
    of a multi-model input. ``per_conformer`` goes through
    ``SFcalculator.calc_fsolvent_batch``, the path the structure-factor reward uses, so
    it inherits that path's batch behavior:
    the density is normalized over the whole partition of (at most 10) models and the
    solvent cutoff comes from the partition's first model. Unlike the reward, which
    weights every conformer's mask by ``1/batch_size``, the masks here are weighted by
    population; the two agree for uniform populations.
    """
    if bulk_solvent not in BULK_SOLVENT_MODES:
        raise ValueError(f"bulk_solvent must be one of {BULK_SOLVENT_MODES}, got {bulk_solvent!r}")
    model_fproteins = []  # one [n_hkl] complex Fprotein_asu per model, at occupancy 1.0
    sfcalculator = None
    reference_hkl = None  # [n_hkl, 3] ASU Miller indices every model must land on
    saved_structure = None
    reference_solventpct = None  # solvent fraction of model 1, estimated from its atoms
    for model, occupancy in enumerate(occupancies):
        sfcalculator = None  # release the previous model's scattering table first
        torch.cuda.empty_cache()
        atom_array = load_structure_for_synthetic_reward(
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
        if save_structure_dir is not None:
            conformer = atomarray_to_gemmi(atom_array, cell, space_group)
            if saved_structure is None:
                saved_structure = conformer
            else:
                conformer[0].name = str(model + 1)
                saved_structure.add_model(conformer[0])
        logger.info(f"Model {model + 1}: {len(atom_array)} atoms at population {occupancy:g}")
        gemmi_structure = atomarray_to_gemmi(atom_array, cell, space_group)
        sfcalculator = build_sfcalculator(
            gemmi_structure, resolution, scattering_factor_mode, device
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
    # The loop always runs: _process_single_row requires one occupancy per model.
    assert sfcalculator is not None
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
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
    bulk_solvent: str = "off",
) -> None:
    """Compute and write the summed structure factors of one ensemble.

    Failures are logged rather than raised so one bad row does not stop a batch.

    Parameters
    ----------
    row
        BatchRowForMTZ naming the multi-model file, with one occupancy value per model
        and optional per-row overrides.
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
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save the processed ensemble as ``<input_stem>_sf_input.cif`` beside
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
    flags, and, unless ``bulk_solvent="off"``, ``Ftotal``/``SIGFtotal``/``PHIFtotal``:
    the layouts ``generate_synthetic_sf.py`` writes without and with
    ``--simulate-solvent-and-scale``.
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
        cell = row.unit_cell if row.unit_cell is not None else gemmi_meta.cell
        space_group = row.space_group if row.space_group is not None else gemmi_meta.spacegroup_hm
        logger.info(f"Ensemble cell {cell.parameters}, space group '{space_group}'")
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
    logger.info(
        f"Summed {len(row.occupancy_values)} models at occupancies "
        f"{row.occupancy_values!r} (bulk solvent {bulk_solvent}) into {output_path}"
    )


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
    strip_hydrogens: bool = False,
    strip_waters: bool = False,
    strip_ligands: bool = False,
    save_structure: bool = False,
    b_factor: float | None = None,
    bulk_solvent: str = "off",
) -> None:
    """Process multiple ensembles from a CSV file in batch mode.

    Parameters
    ----------
    csv_path
        Path to CSV file listing ensembles to process, one multi-model file per row.
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
    strip_hydrogens
        If True, remove hydrogen atoms before computing structure factors.
    strip_waters
        If True, remove water molecules before computing structure factors.
    strip_ligands
        If True, keep only polymer amino-acid atoms (removes ligands and waters).
    save_structure
        If True, save each processed ensemble as mmCIF beside its MTZ.
    b_factor
        Optional isotropic B-factor assigned to every retained atom.
    bulk_solvent
        Bulk-solvent mode: ``"off"``, ``"combined"``, or ``"per_conformer"``.
    """
    from joblib import delayed, Parallel

    rows = load_batch_csv(csv_path)
    effective_n_jobs = resolve_parallel_jobs(device, n_jobs)
    logger.info(f"Processing {len(rows)} ensembles from {csv_path} using {effective_n_jobs} jobs")

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
        Parsed ensemble structure factor configuration.
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
        help="Base directory for relative paths in CSV, not used in single-ensemble mode",
    )

    selection_group = parser.add_argument_group("Selection Options")
    selection_group.add_argument(
        "--selection",
        type=str,
        help="Atom selection applied to every model (e.g., 'chain A and resi 10-50')",
    )

    occupancy_group = parser.add_argument_group("Occupancy Options")
    occupancy_group.add_argument(
        "--occupancy-values",
        type=str,
        help="Colon-separated population per model, summing to 1 (e.g., '0.3:0.7')",
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

    crystal_group = parser.add_argument_group("Crystal Options (single-ensemble mode only)")
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
        help="Save the processed ensemble as <input_stem>_sf_input.cif beside the MTZ",
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
    """Generate ensemble MTZs using command-line arguments."""
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
