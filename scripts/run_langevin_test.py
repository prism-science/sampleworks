"""Smoke test / small sweep for AnnealedLangevinSampler against the CSG baseline.

The model/reward/structure are loaded once and reused across every
--langevin-factor value, since checkpoint loading dominates per-run time
otherwise.

Usage
-----
    pixi run -e protenix python scripts/run_langevin_test.py --langevin-factor 0.2,0.3,0.4,0.5
"""

import argparse
import csv
import dataclasses
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
DATASET_ROOT = Path("/mnt/diffuse-shared/raw/sampleworks/initial_dataset_40_occ_sweeps")


def resolve_inputs(
    protein: str | None, occ: str, dataset_root: Path
) -> tuple[str, str, float, Path]:
    """Return (structure_path, density_path, resolution, output_base) for a run.

    With ``protein=None`` this is the 1VME test resource used by all earlier
    sweeps. Otherwise the row ``{protein}_{occ}`` of the occ-sweep dataset's
    ``proteins.csv`` is used, with its ``/data/inputs`` prefix (the ACTL mount
    point) replaced by ``dataset_root``.
    """
    if protein is None:
        return STRUCTURE, DENSITY, RESOLUTION, OUTPUT_BASE
    name = f"{protein.upper()}_{occ}"
    with open(dataset_root / "proteins.csv") as f:
        row = next((r for r in csv.DictReader(f) if r["name"] == name), None)
    if row is None:
        raise ValueError(f"{name} not found in {dataset_root / 'proteins.csv'}")

    def localize(path: str) -> str:
        return path.replace("/data/inputs", str(dataset_root), 1)

    return (
        localize(row["structure"]),
        localize(row["density"]),
        float(row["resolution"]),
        OUTPUT_BASE / name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langevin-factor",
        type=str,
        default="0.1",
        help="Comma-separated list of langevin_factor values to sweep, e.g. '0.2,0.3,0.4,0.5'. "
        "Pass '' to run only --include-csg.",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="PDB id from the occ-sweep dataset (e.g. 6B8X). Default: the 1VME test resource.",
    )
    parser.add_argument("--occ", type=str, default="0.5occA_0.5occB")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--inverse-temperature", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument(
        "--sde-mode", choices=["legacy", "reverse_sde", "langevin", "ode"], default="legacy"
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=T_START,
        help="Partial-diffusion start fraction. Use 1/NUM_STEPS rather than 0 for from-scratch "
        "runs: at 0, PureGuidance starts from unit-variance noise, which only EDM's churn "
        "rescales to sigma_max.",
    )
    parser.add_argument("--align-xt-to-x0", action="store_true")
    parser.add_argument(
        "--untempered-guidance",
        action="store_true",
        help="Don't apply inverse_temperature to the CSG guidance term (temper_guidance=False).",
    )
    parser.add_argument(
        "--integrate-func",
        choices=["euler_maruyama", "heun"],
        default="euler_maruyama",
        help="LangevinSamplerConfig.integrate_func (Chroma's integrators).",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--label-suffix",
        type=str,
        default="",
        help="Appended to every run label, so a rerun doesn't overwrite earlier output dirs.",
    )
    parser.add_argument(
        "--include-csg",
        action="store_true",
        help="Also run one AF3EDMSampler (CSG) pass with matching settings for comparison.",
    )
    parser.add_argument(
        "--csg-gamma",
        type=str,
        default="0.1",
        help="Comma-separated DataSpaceDPSScaler step sizes for the CSG runs.",
    )
    parser.add_argument(
        "--langevin-gamma",
        type=str,
        default="0.1",
        help="Comma-separated DataSpaceDPSScaler step sizes for the Langevin runs.",
    )
    parser.add_argument(
        "--edm-gamma0",
        type=float,
        default=0.8,
        help="EDMSamplerConfig.gamma_0 (churn) for the CSG runs; 0 disables churn.",
    )
    parser.add_argument(
        "--csg-init-noise",
        choices=["sampler", "sigma"],
        default="sampler",
        help="Partial-diffusion start noise for CSG runs. 'sampler' uses AF3EDMSampler's "
        "churn increment (~1.5 sigma, or 0 when --edm-gamma0 0); 'sigma' uses sigma, "
        "matching AnnealedLangevinSampler.",
    )
    return parser.parse_args()


class SigmaInitNoise:
    """Delegate to a sampler, but start partial diffusion from noise of std sigma.

    PureGuidance scales its initial noise by the ``noise_scale`` of the context
    for step ``starting_step - 1``. AF3EDMSampler reports its churn increment
    there (~1.5 sigma by default, 0 without churn), whereas
    AnnealedLangevinSampler reports sigma. Only that one context is changed;
    every sampling-step context is passed through untouched.
    """

    def __init__(self, sampler: AF3EDMSampler, init_step_index: int):
        self.sampler = sampler
        self.init_step_index = init_step_index

    def __getattr__(self, name: str):
        return getattr(self.sampler, name)

    def get_context_for_step(self, step_index: int, schedule):
        """Return the wrapped sampler's context, with noise_scale=sigma at the init index."""
        context = self.sampler.get_context_for_step(step_index, schedule)
        if step_index == self.init_step_index:
            context = dataclasses.replace(context, noise_scale=schedule.sigma_tm[step_index])
        return context


def main() -> None:
    args = parse_args()
    factors = [float(v) for v in args.langevin_factor.split(",") if v.strip()]
    csg_gammas = [float(v) for v in args.csg_gamma.split(",") if v.strip()]
    langevin_gammas = [float(v) for v in args.langevin_gamma.split(",") if v.strip()]
    structure_path, density_path, resolution, output_base = resolve_inputs(
        args.protein, args.occ, args.dataset_root
    )
    logger.info(f"Structure {structure_path}, map {density_path} @ {resolution} A")

    device = try_gpu()
    _, model = get_model_and_device(
        device_str=str(device), model_checkpoint_path=None, model_type="protenix"
    )

    reward, structure = get_reward_function_and_structure(
        density=density_path,
        device=device,
        em=False,
        loss_order=2,
        resolution=resolution,
        structure_path=structure_path,
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

    def run_one(
        label: str, sampler, scaler: DataSpaceDPSScaler = step_scaler
    ) -> tuple[str, float, float, float]:
        guidance = PureGuidance(
            ensemble_size=ENSEMBLE_SIZE,
            num_steps=args.num_steps,
            t_start=args.t_start,
            guidance_t_start=0.0,
        )
        if args.num_steps != NUM_STEPS:
            label = f"{label}_steps{args.num_steps}"
        if args.t_start != T_START:
            label = f"{label}_tstart{args.t_start:g}"
        output = guidance.sample(
            structure=structure,
            model=model,
            sampler=sampler,
            step_scaler=scaler,
            reward=reward,
        )
        losses = [loss_value for loss_value in output.losses if loss_value is not None]
        initial_loss, final_loss = losses[0], losses[-1]
        final_rscc = compute_rscc(output.final_state)
        logger.info(
            f"{label}: Initial {initial_loss:.6f} -> Final {final_loss:.6f} "
            f"(reduction {initial_loss - final_loss:.6f}); RSCC={final_rscc:.4f}"
        )

        output_dir = output_base / label
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(output.final_state.detach().cpu(), output_dir / "final_state.pt")
        with open(output_dir / "losses.txt", "w") as f:
            f.write("step,loss\n")
            for i, loss_value in enumerate(losses):
                f.write(f"{i},{loss_value}\n")

        return label, initial_loss, final_loss, final_rscc

    results: list[tuple[str, float, float, float]] = []

    for rep in range(args.repeats):
        rep_suffix = f"{args.label_suffix}_rep{rep}" if args.repeats > 1 else args.label_suffix

        if args.include_csg:
            for csg_gamma in csg_gammas:
                csg_sampler = AF3EDMSampler(
                    EDMSamplerConfig(device=str(device), gamma_0=args.edm_gamma0)
                )
                label = "csg"
                if csg_gamma != 0.1:
                    label = f"{label}_g{csg_gamma:g}"
                if args.edm_gamma0 != 0.8:
                    label = f"{label}_churn{args.edm_gamma0:g}"
                if args.csg_init_noise == "sigma":
                    label = f"{label}_initsigma"
                    csg_sampler = SigmaInitNoise(
                        csg_sampler, int(args.t_start * args.num_steps) - 1
                    )
                csg_scaler = DataSpaceDPSScaler(step_size=csg_gamma, gradient_normalization=True)
                results.append(run_one(f"{label}{rep_suffix}", csg_sampler, csg_scaler))

        for factor in factors:
            for langevin_gamma in langevin_gammas:
                langevin_sampler = AnnealedLangevinSampler(
                    LangevinSamplerConfig(
                        device=str(device),
                        inverse_temperature=args.inverse_temperature,
                        langevin_factor=factor,
                        sde_mode=args.sde_mode,
                        align_xt_to_x0=args.align_xt_to_x0,
                        temper_guidance=not args.untempered_guidance,
                        integrate_func=args.integrate_func,
                    )
                )
                label = f"factor{factor}"
                if args.inverse_temperature != 1.0:
                    label = f"{label}_invT{args.inverse_temperature}"
                if args.untempered_guidance:
                    label = f"{label}_untempG"
                if langevin_gamma != 0.1:
                    label = f"{label}_g{langevin_gamma:g}"
                if args.integrate_func != "euler_maruyama":
                    label = f"{label}_heun"
                if args.sde_mode != "legacy" or args.align_xt_to_x0:
                    label = f"{args.sde_mode}{'_alignx0' if args.align_xt_to_x0 else ''}_{label}"
                langevin_scaler = DataSpaceDPSScaler(
                    step_size=langevin_gamma, gradient_normalization=True
                )
                results.append(
                    run_one(f"{label}{rep_suffix}", langevin_sampler, langevin_scaler)
                )

    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "summary.csv", "w") as f:
        f.write("label,initial_loss,final_loss,reduction,rscc\n")
        for label, initial_loss, final_loss, final_rscc in results:
            f.write(
                f"{label},{initial_loss},{final_loss},{initial_loss - final_loss},{final_rscc}\n"
            )
    logger.info(f"Saved sweep results to {output_base}/")


if __name__ == "__main__":
    main()
