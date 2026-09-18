"""Smoke test / small sweep for AnnealedLangevinSampler against the CSG baseline.

Not wired into the sampleworks-guidance CLI -- this constructs the same
model/reward/scaler as the CSG baseline run directly and swaps in
``AnnealedLangevinSampler`` in place of ``AF3EDMSampler``, so the loss curve
can be compared to the existing baseline numbers (Initial 0.025783 -> Final
0.007554 for the full-parameter CSG run on 1VME). See
``core/samplers/langevin.py`` for the sampler itself and its docstring for
what is/isn't a faithful port of Chroma's Annealed Langevin SDE.

The model/reward/structure are loaded once and reused across every
``--langevin-factor`` value, since checkpoint loading dominates per-run time
otherwise.

Usage
-----
    pixi run -e protenix python scripts/run_langevin_test.py --langevin-factor 0.2,0.3,0.4,0.5
"""

import argparse
from pathlib import Path

import torch
from loguru import logger

from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
from sampleworks.eval.metrics import rscc
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
)
from sampleworks.utils.torch_utils import try_gpu


STRUCTURE = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY = "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION = 1.8
NUM_STEPS = 50
ENSEMBLE_SIZE = 2
T_START = 0.6  # matches --partial-diffusion-step 30 of 50 in the quick CSG test run
OUTPUT_BASE = Path("output/langevin_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langevin-factor",
        type=str,
        default="0.1",
        help="Comma-separated list of langevin_factor values to sweep, e.g. '0.2,0.3,0.4,0.5'",
    )
    parser.add_argument("--inverse-temperature", type=float, default=1.0)
    parser.add_argument(
        "--include-csg",
        action="store_true",
        help="Also run one AF3EDMSampler (CSG) pass with matching settings for comparison.",
    )
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

    step_scaler = DataSpaceDPSScaler(step_size=0.1, gradient_normalization=True)

    # Build reward_inputs (elements/b_factors/occupancies) once, the same way
    # PureGuidance.sample() does internally, so we can compute a whole-map RSCC
    # for each sweep result's final_state without re-running the model.
    features = model.featurize(structure)
    prior_coords = torch.as_tensor(
        model.initialize_from_prior(batch_size=ENSEMBLE_SIZE, features=features)
    )
    processed_structure = process_structure_to_trajectory_input(
        structure=structure,
        coords_from_prior=prior_coords,
        features=features,
        ensemble_size=ENSEMBLE_SIZE,
    )
    reward_inputs = processed_structure.to_reward_inputs(device=device)

    def compute_rscc(final_state: torch.Tensor) -> float:
        """Whole-map RSCC between the ensemble's density and the target map.

        Not the paper's per-segment tight-extraction RSCC (Appendix F) --
        this correlates the full map, since no altloc-segment selection is
        loaded for this smoke test. See scripts/eval/rscc_grid_search_script.py
        for the segment-restricted version this simplifies.
        """
        with torch.no_grad():
            density = reward.transformer(
                coordinates=final_state,
                elements=reward_inputs.elements,
                b_factors=reward_inputs.b_factors,
                occupancies=reward_inputs.occupancies,
            ).sum(0)
        target_array = reward.transformer.xmap.array
        target_np = (
            target_array.detach().cpu().numpy()
            if torch.is_tensor(target_array)
            else target_array
        )
        return float(rscc(density.detach().cpu().numpy(), target_np))

    def run_one(label: str, sampler) -> tuple[str, float, float, float]:
        guidance = PureGuidance(
            ensemble_size=ENSEMBLE_SIZE, num_steps=NUM_STEPS, t_start=T_START, guidance_t_start=0.0
        )
        output = guidance.sample(
            structure=structure,
            model=model,
            sampler=sampler,
            step_scaler=step_scaler,
            reward=reward,
        )
        losses = [loss_value for loss_value in output.losses if loss_value is not None]
        initial_loss, final_loss = losses[0], losses[-1]
        final_rscc = compute_rscc(output.final_state)
        logger.info(
            f"{label}: Initial {initial_loss:.6f} -> Final {final_loss:.6f} "
            f"(reduction {initial_loss - final_loss:.6f}); RSCC={final_rscc:.4f}"
        )

        output_dir = OUTPUT_BASE / label
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(output.final_state.detach().cpu(), output_dir / "final_state.pt")
        with open(output_dir / "losses.txt", "w") as f:
            f.write("step,loss\n")
            for i, loss_value in enumerate(losses):
                f.write(f"{i},{loss_value}\n")

        return label, initial_loss, final_loss, final_rscc

    results: list[tuple[str, float, float, float]] = []

    if args.include_csg:
        csg_sampler = AF3EDMSampler(EDMSamplerConfig(device=str(device)))
        results.append(run_one("csg", csg_sampler))

    for factor in factors:
        langevin_sampler = AnnealedLangevinSampler(
            LangevinSamplerConfig(
                device=str(device),
                inverse_temperature=args.inverse_temperature,
                langevin_factor=factor,
            )
        )
        results.append(run_one(f"factor{factor}", langevin_sampler))

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BASE / "summary.csv", "w") as f:
        f.write("label,initial_loss,final_loss,reduction,rscc\n")
        for label, initial_loss, final_loss, final_rscc in results:
            f.write(
                f"{label},{initial_loss},{final_loss},{initial_loss - final_loss},{final_rscc}\n"
            )
    logger.info(f"Saved sweep results to {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
