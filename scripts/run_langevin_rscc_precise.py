"""Paper-aligned RSCC sweep for AnnealedLangevinSampler vs the CSG baseline.

Unlike ``run_langevin_test.py``'s whole-map RSCC, this restricts the
correlation to the tight (2.0 A padding) voxel region around the altloc
selection for the target structure, exactly matching the methodology in
``scripts/eval/rscc_grid_search_script.py`` (and the paper's Appendix F):
voxels within 2.0 A of any atom center in the segment.

The selection ("chain A and resi 326-339") comes from
``src/sampleworks/data/protein_configs.csv``'s 1vme row.

Usage
-----
    pixi run -e protenix python scripts/run_langevin_rscc_precise.py \
        --langevin-factor 0.0,0.1,0.2,0.3,0.4,0.5 --include-csg
"""

import argparse
import copy
from pathlib import Path

import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import stack
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
from sampleworks.eval.metrics import rscc
from sampleworks.utils.atom_array_utils import apply_selection, parse_structure
from sampleworks.utils.density_utils import build_density_transformer, run_density_transformer
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
)
from sampleworks.utils.torch_utils import try_gpu


STRUCTURE = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION = 1.8
SELECTION = "chain A and resi 326-339"  # from src/sampleworks/data/protein_configs.csv, 1vme row
SELECTION_PADDING = 2.0  # matches DEFAULT_SELECTION_PADDING / paper Appendix F
NUM_STEPS = 50
ENSEMBLE_SIZE = 2
T_START = 0.6
OUTPUT_BASE = Path("output/langevin_rscc_precise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langevin-factor", type=str, default="0.1")
    parser.add_argument("--inverse-temperature", type=float, default=1.0)
    parser.add_argument("--include-csg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factors = [float(v) for v in args.langevin_factor.split(",")]

    device = try_gpu()
    _, model = get_model_and_device(
        device_str=str(device), model_checkpoint_path=None, model_type="protenix"
    )

    reward, structure = get_reward_function_and_structure(
        density=DENSITY,
        device=device,
        em=False,
        loss_order=2,
        resolution=RESOLUTION,
        structure_path=STRUCTURE,
    )

    # Reference selection coordinates, used only to pick which voxels to look
    # at -- these come from the deposited/reference structure, not the model.
    reference_atom_array = ensure_atom_array_stack(parse_structure(STRUCTURE)["asym_unit"])[0]
    sel_atoms = apply_selection(reference_atom_array, SELECTION)
    sel_coords = sel_atoms.coord

    # Target map (raw qfit XMap, for extract_tight) and its tight-extracted
    # reference density -- computed once, reused for every run.
    base_xmap = XMap.fromfile(DENSITY, resolution=RESOLUTION)
    _, extracted_target = base_xmap.extract_tight(sel_coords, padding=SELECTION_PADDING)
    logger.info(f"Selection '{SELECTION}': {len(sel_atoms)} atoms, {extracted_target.shape[0]} voxels")

    density_transformer, _ = build_density_transformer(base_xmap, em_mode=False, device=device)

    def compute_precise_rscc(final_state: torch.Tensor, model_atom_array) -> float:
        """RSCC restricted to the 2.0 A tight region around SELECTION."""
        ensemble_size = final_state.shape[0]
        ensemble_array = stack([model_atom_array.copy() for _ in range(ensemble_size)])
        ensemble_array.coord = final_state.detach().cpu().numpy()

        computed_density = run_density_transformer(density_transformer, ensemble_array)
        computed_xmap = copy.copy(base_xmap)
        computed_xmap.array = computed_density.cpu().numpy()

        _, extracted_computed = computed_xmap.extract_tight(sel_coords, padding=SELECTION_PADDING)
        return float(rscc(extracted_target, extracted_computed))

    step_scaler = DataSpaceDPSScaler(step_size=0.1, gradient_normalization=True)

    def run_one(label: str, sampler) -> tuple[str, float, float, float]:
        guidance = PureGuidance(
            ensemble_size=ENSEMBLE_SIZE, num_steps=NUM_STEPS, t_start=T_START, guidance_t_start=0.0
        )
        output = guidance.sample(
            structure=structure, model=model, sampler=sampler, step_scaler=step_scaler, reward=reward
        )
        losses = [v for v in output.losses if v is not None]
        initial_loss, final_loss = losses[0], losses[-1]

        model_atom_array = (output.metadata or {}).get("model_atom_array")
        if model_atom_array is None:
            # No atom-count mismatch this run; the structure's own atom array
            # is already in model atom order.
            model_atom_array = reference_atom_array
        final_rscc = compute_precise_rscc(output.final_state, model_atom_array)

        logger.info(
            f"{label}: Initial {initial_loss:.6f} -> Final {final_loss:.6f} "
            f"(reduction {initial_loss - final_loss:.6f}); tight RSCC={final_rscc:.4f}"
        )
        return label, initial_loss, final_loss, final_rscc

    results: list[tuple[str, float, float, float]] = []
    if args.include_csg:
        results.append(run_one("csg", AF3EDMSampler(EDMSamplerConfig(device=str(device)))))
    for factor in factors:
        sampler = AnnealedLangevinSampler(
            LangevinSamplerConfig(
                device=str(device), inverse_temperature=args.inverse_temperature, langevin_factor=factor
            )
        )
        results.append(run_one(f"factor{factor}", sampler))

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BASE / "summary.csv", "w") as f:
        f.write("label,initial_loss,final_loss,reduction,tight_rscc\n")
        for label, initial_loss, final_loss, final_rscc in results:
            f.write(f"{label},{initial_loss},{final_loss},{initial_loss - final_loss},{final_rscc}\n")
    logger.info(f"Saved results to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
