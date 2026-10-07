"""Convert saved final_state.pt files (from run_langevin_test.py) into viewable CIFs.

Rebuilds the model atom-array template the same way PureGuidance.sample()
does, without re-running the sampler, writes each saved ensemble into it and
saves one multi-model CIF per run (one model per ensemble member). Coordinates
are already in the input/map frame, so the CIFs overlay directly on the
target map in PyMOL/ChimeraX.

Usage
-----
    pixi run -e protenix python scripts/convert_final_state_to_cif.py --protein 6B8X \
        --out-dir structures/6B8X output/langevin_sweep/6B8X_0.5occA_0.5occB/csg_rep0_steps500
"""

import argparse
from pathlib import Path

import torch
from biotite.structure import stack
from biotite.structure.io.pdbx import CIFFile, set_structure
from run_langevin_test import DATASET_ROOT, resolve_inputs
from sampleworks.utils.guidance_script_utils import get_model_and_device, load_guidance_structure
from sampleworks.utils.structure_utils import process_structure_to_trajectory_input
from sampleworks.utils.torch_utils import try_gpu


def parse_args() -> argparse.Namespace:
    """Parse run directories and the input they were sampled from."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="PDB id from the occ-sweep dataset. Default: the 1VME test resource.",
    )
    parser.add_argument("--occ", type=str, default="0.5occA_0.5occB")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    """Write ``<out_dir>/<run_dir name>.cif`` for every run directory."""
    args = parse_args()
    structure_path, density_path, resolution, _ = resolve_inputs(
        args.protein, args.occ, args.dataset_root
    )

    device = try_gpu()
    _, model = get_model_and_device(
        device_str=str(device), model_checkpoint_path=None, model_type="protenix"
    )
    structure = load_guidance_structure(structure_path)
    features = model.featurize(structure)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for run_dir in args.run_dirs:
        final_state = torch.load(run_dir / "final_state.pt", map_location="cpu")  # [ens, atoms, 3]
        ensemble_size = final_state.shape[0]
        prior_coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=ensemble_size, features=features)
        )
        processed_structure = process_structure_to_trajectory_input(
            structure=structure,
            coords_from_prior=prior_coords,
            features=features,
            ensemble_size=ensemble_size,
        )
        template = processed_structure.reward_atom_array

        ensemble_array = stack([template.copy() for _ in range(ensemble_size)])
        ensemble_array.coord = final_state.numpy()

        cif_file = CIFFile()
        set_structure(cif_file, ensemble_array)
        out_path = args.out_dir / f"{run_dir.name}.cif"
        cif_file.write(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
