"""Tight (paper-style) RSCC for final_state.pt files saved by run_langevin_test.py.

Scores already-sampled ensembles instead of re-running the sampler, so the
tight and whole-map RSCC numbers describe the same samples. Density is
computed with the same reward inputs (elements, B-factors, occupancies) that
guidance used, on the target map's grid. Each altloc segment is scored on the
voxels within 2.0 A of its atoms in *either* altloc, using the same masking
helpers as scripts/eval/rscc_grid_search_script.py.

Segments come from the paper's ``classify_altloc_selections.csv`` (791
single-conformer-range segments over the 40 proteins); the combined
per-protein expression in that file is skipped.

Usage
-----
    pixi run -e protenix python scripts/eval_saved_tight_rscc.py --protein 6B8X \
        output/langevin_sweep/6B8X_0.5occA_0.5occB/*_rep*
"""

import argparse
import copy
import csv
import statistics
from pathlib import Path

import torch
from loguru import logger
from run_langevin_test import DATASET_ROOT, resolve_inputs

from sampleworks.eval.constants import DEFAULT_SELECTION_PADDING
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import (
    get_reference_structure_coords,
    process_structure_to_trajectory_input,
)
from sampleworks.utils.density_utils import build_density_transformer
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
)
from sampleworks.utils.torch_utils import try_gpu


SELECTIONS_CSV = Path("/mnt/diffuse-shared/raw/sampleworks/classify_altloc_selections.csv")
SELECTION = "chain A and resi 326-339"  # src/sampleworks/data/protein_configs.csv, 1vme row
ENSEMBLE_SIZE = 2


def load_segments(protein: str | None, selections_csv: Path) -> list[str]:
    """Return the altloc segment selections to score for ``protein``.

    ``protein=None`` means the 1VME test resource, scored on the single
    segment from ``src/sampleworks/data/protein_configs.csv``.
    """
    if protein is None:
        return [SELECTION]
    with open(selections_csv) as f:
        row = next((r for r in csv.DictReader(f) if r["protein"] == protein.upper()), None)
    if row is None:
        raise ValueError(f"{protein} not found in {selections_csv}")
    return [s.strip() for s in row["selection"].split(";") if s.strip() and "==" not in s]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--protein", type=str, default=None)
    parser.add_argument("--occ", type=str, default="0.5occA_0.5occB")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--selections-csv", type=Path, default=SELECTIONS_CSV)
    return parser.parse_args()


def main() -> None:
    """Print whole-map RSCC and per-segment tight RSCC summaries for each run directory."""
    args = parse_args()
    structure_path, density_path, resolution, output_base = resolve_inputs(
        args.protein, args.occ, args.dataset_root
    )
    segments = load_segments(args.protein, args.selections_csv)

    device = try_gpu()
    _, model = get_model_and_device(
        device_str=str(device), model_checkpoint_path=None, model_type="protenix"
    )
    _, structure = get_reward_function_and_structure(
        density=density_path, device=device, em=False, loss_order=2,
        resolution=resolution, structure_path=structure_path,
    )
    features = model.featurize(structure)
    prior_coords = torch.as_tensor(
        model.initialize_from_prior(batch_size=ENSEMBLE_SIZE, features=features)
    )
    processed_structure = process_structure_to_trajectory_input(
        structure=structure, coords_from_prior=prior_coords, features=features,
        ensemble_size=ENSEMBLE_SIZE,
    )
    reward_inputs = processed_structure.to_reward_inputs(device=device)

    # Concrete (occupancy-independent) patterns: both altloc lookups in
    # get_reference_structure_coords resolve to the same multi-altloc file,
    # whose A and B coordinates are all kept for the mask.
    structure_file = Path(structure_path)
    protein_config = ProteinConfig(
        protein=args.protein or "1VME",
        base_map_dir=structure_file.parent,
        selection=segments,
        resolution=resolution,
        map_pattern=Path(density_path).name,
        structure_pattern=structure_file.name,
    )
    segment_coords = get_reference_structure_coords(protein_config, protein_config.protein) or {}
    missing = [s for s in segments if s not in segment_coords]
    if missing:
        logger.warning(f"No reference coords for {len(missing)} segment(s): {missing}")

    # Same loader as rscc_grid_search_script.py: expands to the canonical unit cell,
    # which extract_tight requires.
    base_xmap = protein_config.load_map(Path(density_path), resolution=resolution)
    if base_xmap is None:
        raise ValueError(f"Failed to load map {density_path}")
    transformer, _ = build_density_transformer(base_xmap, em_mode=False, device=device)
    extracted_targets = {
        s: base_xmap.extract_tight(coords, padding=DEFAULT_SELECTION_PADDING)[1]
        for s, coords in segment_coords.items()
    }

    per_segment_rows = []
    print("label,whole_rscc,median_tight_rscc,frac_ge_0.8,n_segments")
    for run_dir in args.run_dirs:
        final_state = torch.load(run_dir / "final_state.pt", map_location=device)  # [ens, atoms, 3]
        with torch.no_grad():
            density = transformer(
                coordinates=final_state,
                elements=reward_inputs.elements,
                b_factors=reward_inputs.b_factors,
                occupancies=reward_inputs.occupancies,
            ).sum(0)
        computed_xmap = copy.copy(base_xmap)
        computed_xmap.array = density.cpu().numpy()

        tight = []
        for segment, coords in segment_coords.items():
            _, extracted_computed = computed_xmap.extract_tight(
                coords, padding=DEFAULT_SELECTION_PADDING
            )
            value = float(rscc(extracted_targets[segment], extracted_computed))
            tight.append(value)
            per_segment_rows.append((run_dir.name, segment, value))

        whole = float(rscc(computed_xmap.array, base_xmap.array))
        frac = sum(v >= 0.8 for v in tight) / len(tight)
        print(f"{run_dir.name},{whole:.4f},{statistics.median(tight):.4f},{frac:.3f},{len(tight)}")

    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "tight_rscc_segments.csv", "w") as f:
        f.write("label,selection,tight_rscc\n")
        for label, segment, value in per_segment_rows:
            f.write(f'{label},"{segment}",{value}\n')
    logger.info(f"Per-segment RSCC written to {output_base / 'tight_rscc_segments.csv'}")


if __name__ == "__main__":
    main()
