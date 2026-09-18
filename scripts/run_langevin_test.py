"""First smoke test for AnnealedLangevinSampler against the CSG baseline.

Not wired into the sampleworks-guidance CLI -- this constructs the same
model/reward/scaler as the CSG baseline run directly and swaps in
``AnnealedLangevinSampler`` in place of ``AF3EDMSampler``, so the loss curve
can be compared to the existing baseline numbers (Initial 0.025783 -> Final
0.007554 for the full-parameter CSG run on 1VME). See
``core/samplers/langevin.py`` for the sampler itself and its docstring for
what is/isn't a faithful port of Chroma's Annealed Langevin SDE.

Usage
-----
    pixi run -e protenix python scripts/run_langevin_test.py
"""

from pathlib import Path

import torch
from loguru import logger

from sampleworks.core.samplers.langevin import AnnealedLangevinSampler, LangevinSamplerConfig
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
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
OUTPUT_DIR = Path("output/langevin_test")


def main() -> None:
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
    guidance = PureGuidance(
        ensemble_size=ENSEMBLE_SIZE, num_steps=NUM_STEPS, t_start=T_START, guidance_t_start=0.0
    )
    sampler = AnnealedLangevinSampler(
        LangevinSamplerConfig(device=str(device), inverse_temperature=1.0, langevin_factor=0.5)
    )

    output = guidance.sample(
        structure=structure, model=model, sampler=sampler, step_scaler=step_scaler, reward=reward
    )

    losses = [loss_value for loss_value in output.losses if loss_value is not None]
    logger.info(f"Initial loss: {losses[0]:.6f}")
    logger.info(f"Final loss:   {losses[-1]:.6f}")
    logger.info(f"Loss reduction: {losses[0] - losses[-1]:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(output.final_state.detach().cpu(), OUTPUT_DIR / "final_state.pt")
    with open(OUTPUT_DIR / "losses.txt", "w") as f:
        f.write("step,loss\n")
        for i, loss_value in enumerate(losses):
            f.write(f"{i},{loss_value}\n")
    logger.info(f"Saved final_state.pt and losses.txt to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
