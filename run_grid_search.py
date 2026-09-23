"""Grid search script for running trials across models, scalers, and parameters."""

from __future__ import annotations

import argparse
import concurrent
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from loguru import logger as log
from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import (
    add_latent_opt_args,
    GuidanceConfig,
    JobConfig,
    JobResult,
)
from sampleworks.utils.protein_input import ProteinInput


@dataclass
class GridSearchConfig:
    """Serializable summary of the grid-search dimensions and output location."""

    model_name: str
    scalers: list[str]
    ensemble_sizes: list[int]
    gradient_weights: list[float]
    gd_steps: list[int]
    method: str
    proteins_file: str
    output_dir: str


def get_job_status(job: JobConfig) -> str:
    """
    Check the status of a job by inspecting its log file.

    Returns:
        'success': Job completed successfully (has "Final loss:" in log)
        'failed': Job ran but failed (has errors/traceback in log or exit != 0)
        'not_run': Job has not been executed yet (no log file)
    """
    if not os.path.exists(job.log_path):
        return "not_run"

    try:
        with open(job.log_path) as f:
            log_content = f.read()

        has_error = (
            "Traceback" in log_content or "AssertionError" in log_content or "Error:" in log_content
        )
        if has_error:
            return "failed"

        if "Final loss:" in log_content:
            return "success"

        return "failed"
    except Exception as e:
        log.warning(f"Error reading log file {job.log_path}: {e}")
        return "failed"


def _gpu_indices_from_torch() -> list[str] | None:
    """Return visible CUDA ordinals using PyTorch when it is importable.

    Returns
    -------
    list of str or None
        Visible local CUDA ordinals. ``None`` means PyTorch is unavailable or
        CUDA discovery failed before returning a device count.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        if not torch.cuda.is_available():
            return []
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:
        log.debug(f"PyTorch CUDA discovery failed: {exc}")
        return None


def _gpu_indices_from_nvidia_smi() -> list[str] | None:
    """Return visible CUDA ordinals using ``nvidia-smi`` as a fallback.

    Returns
    -------
    list of str or None
        GPU ordinals reported by ``nvidia-smi``. ``None`` means the command is
        absent or failed.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]


def _discover_gpu_indices() -> list[str] | None:
    """Return visible CUDA ordinals from Python first, then ``nvidia-smi``.

    Returns
    -------
    list of str or None
        Visible GPU ordinals, or ``None`` when discovery is unavailable.
    """
    torch_indices = _gpu_indices_from_torch()
    if torch_indices is not None:
        return torch_indices
    return _gpu_indices_from_nvidia_smi()


def detect_gpus() -> list[str]:
    """Return CUDA GPU identifiers visible to this grid-search process.

    ``CUDA_VISIBLE_DEVICES`` wins when set because CUDA remaps those entries to
    local process ordinals. Explicit CUDA "no device" sentinel values return an
    empty list. Otherwise, ``nvidia-smi`` is used as a best-effort discovery
    mechanism and ``["0"]`` is returned as a CPU/test fallback.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cuda_visible_key = cuda_visible.lower()
    if cuda_visible_key in {"none", "void", "nodevfiles"}:
        return []
    if cuda_visible_key == "all":
        return _discover_gpu_indices() or ["0"]
    if cuda_visible and cuda_visible_key != "all":
        gpus = [g.strip() for g in cuda_visible.split(",") if g.strip()]
        visible = _discover_gpu_indices()
        if visible and all(g.isdigit() for g in gpus + visible):
            missing = sorted(set(gpus).difference(visible), key=int)
            if missing:
                raise ValueError(
                    "CUDA_VISIBLE_DEVICES references GPUs that are not visible "
                    f"in this container: {missing}. Visible GPUs: {visible}. "
                    "Check the preset jobs.*.gpus values for this pod size."
                )
        return gpus
    discovered = _discover_gpu_indices()
    if discovered is not None:
        return discovered
    return ["0"]


# Every StructurePredictor must appear here; a missing entry only fails inside the
# worker subprocess, long after the grid has been generated.
MODEL_PIXI_ENVS: dict[StructurePredictor, str] = {
    StructurePredictor.BOLTZ_1: "boltz",
    StructurePredictor.BOLTZ_2: "boltz",
    StructurePredictor.PROTENIX: "protenix",
    StructurePredictor.RF3: "rf3",
    StructurePredictor.PROTPARDELLE: "protpardelle",
}


def get_pixi_env(model: str) -> str:
    """Return the pixi environment name needed to run a model family.

    Parameters
    ----------
    model : str
        Structure predictor name, e.g. ``boltz2`` or ``protpardelle``.

    Returns
    -------
    str
        Pixi environment name that provides the model's dependencies.

    Raises
    ------
    ValueError
        If the model is not a known ``StructurePredictor`` or has no mapped
        pixi environment.
    """
    try:
        return MODEL_PIXI_ENVS[StructurePredictor(model)]
    except (ValueError, KeyError):
        valid_options = [m.value for m in MODEL_PIXI_ENVS]
        raise ValueError(f"Unknown model: {model}. Valid options are: {valid_options}") from None


def build_args_for_process_pool(
    job: JobConfig, args: argparse.Namespace, device_num: int | None = None
) -> GuidanceConfig:
    """Convert a grid-search job into the picklable guidance config for a worker."""
    guidance_config = GuidanceConfig(
        protein=job.protein,
        structure=job.structure_path,
        density=job.density_path,
        model_name=job.model_name,
        guidance_type=job.scaler,
        log_path=job.log_path,
        output_dir=job.output_dir,
        loss_order=args.loss_order,
        partial_diffusion_step=args.partial_diffusion_step,
        guidance_start=args.guidance_start,
        resolution=job.resolution,
        device=f"cuda:{device_num}" if device_num is not None else "",
        gradient_normalization=args.gradient_normalization,
        augmentation=args.augmentation,
        align_to_input=args.align_to_input,
        recycling_steps=args.recycling_steps,
        num_diffusion_steps=args.num_diffusion_steps,
    )
    # given model_type and guidance_type, the GuidanceConfig class will set itself up
    # with defaults for remaining required args, but we want to set them further here.
    guidance_config.populate_config_for_guidance_type(job, args)
    return guidance_config


def worker_log_path_for(job_queue_path: str) -> str:
    """Return the path a worker redirects its subprocess output to.

    Derived in one place so the collector's error messages always name the
    file ``run_guidance_queue_script`` actually wrote.

    Parameters
    ----------
    job_queue_path : str
        Path to the worker's pickled job queue.

    Returns
    -------
    str
        Sibling ``.log`` path for that worker's subprocess output.
    """
    return job_queue_path.replace(".pkl", ".log")


def run_grid_search(
    jobs: list[JobConfig],
    gpus: list[str],
    args: argparse.Namespace,
    job_statuses: dict[int, str] | None = None,
) -> list[JobResult]:
    """
    Replacing run_job and run_grid_search, avoiding model reloads
    of the model.
    Args:
        jobs: generated by generate_and_filter_jobs, a list of JobConfig objects
        gpus: the available GPUs to run jobs on, really we only use the length of the list
        args: command-line arguments to this script, used to pass some on to jobs.
        job_statuses: a dictionary mapping job IDs to their statuses,
            also generated by generate_and_filter_jobs.

    Returns:
        list of JobResult objects, one for each job run, primarily to track success/failure
    """
    results: list[JobResult] = []
    successful = 0
    failed = 0

    # Workers per GPU (--jobs-per-gpu) lets two jobs share a card, which is worthwhile because a
    # single job leaves the GPU idle while it featurizes and writes output. Two constraints shape
    # the worker count:
    #   never more workers than jobs -- a surplus worker gets an empty queue, and the
    #     worker_job_queues[worker_num][0] lookup below then raises IndexError before any model
    #     loads. Easy to hit once jobs_per_gpu multiplies the count.
    #   keep it a whole multiple of the GPU count, so every card carries the same load. Clamping
    #     to the job count alone leaves e.g. 5 workers on 4 GPUs: one card runs two jobs
    #     concurrently while the other three run one and then idle.
    max_workers = min(len(gpus) * args.jobs_per_gpu, len(jobs))
    if max_workers > len(gpus):
        max_workers -= max_workers % len(gpus)
    gpus_used = min(max_workers, len(gpus))
    log.info(
        f"Running {len(jobs)} jobs with {max_workers} parallel workers "
        f"({gpus_used} GPU(s) x {max_workers // gpus_used} jobs/GPU)"
    )

    # Divide the job among the workers. The worker index is no longer the GPU ordinal, so the
    # device wraps with i % len(gpus): workers 0..7 across 4 GPUs pair up as 0,1,2,3,0,1,2,3.
    # Passing i directly would ask for cuda:4..cuda:7 on a 4-GPU box and fail immediately.
    worker_job_queues = [
        [build_args_for_process_pool(j, args, i % len(gpus)) for j in jobs[i::max_workers]]
        for i in range(max_workers)
    ]
    # we'll pickle each job queue separately and then execute each job queue in a separate process
    # in principle we could just pass the job queue directly to the worker_wrapper function, but
    # this keeps a record of what we did, which may be useful for debugging.
    job_queue_paths = []
    for wjq in worker_job_queues:
        wjq_path = os.path.join(args.output_dir, f"wjq_{id(wjq)}.pkl")
        log.info(f"Pickling worker job queue to {wjq_path}")
        job_queue_paths.append(wjq_path)
        with open(wjq_path, "wb") as f:
            pickle.dump(wjq, f)

    if args.dry_run:
        log.info(f"[DRY-RUN] Running {len(jobs)} jobs with {max_workers} parallel workers")
        log.info(f"[DRY-RUN] Job queue paths: {job_queue_paths}")
        return results

    # Clean up output directories if they already exist and have failed previously:
    for i, job in enumerate(jobs):
        clean_output = False
        if job_statuses is not None:
            clean_output = job_statuses.get(id(job), "not_run") != "not_run"
        if clean_output and os.path.exists(job.output_dir):
            log.info(f"Cleaning existing output directory: {job.output_dir}")
            shutil.rmtree(job.output_dir)

    # TODO this approach works, but I think it may be more efficient to actually call a
    #  script that runs the jobs (a script that basically only calls run_guidance_job_queue
    #  to avoid pickling objects.
    with ProcessPoolExecutor(
        max_workers=max_workers,  # TODO: may need to tune this or make a flag.
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {}

        for worker_num, job_queue_path in enumerate(job_queue_paths):
            model = worker_job_queues[worker_num][0].model_name
            future = executor.submit(
                run_guidance_queue_script,
                job_queue_path,
                model,
                worker_num,
                gpus,
            )
            futures[future] = job_queue_path

        for completed in concurrent.futures.as_completed(futures):  # ty: ignore
            job_queue_path = futures[completed]
            # Same file the worker redirects its subprocess output to; these
            # messages exist to point the reader at it.
            worker_log_path = worker_log_path_for(job_queue_path)
            # The worker raises before it can write results, so surface its
            # traceback here rather than reporting the missing results file.
            worker_error = completed.exception()
            if worker_error is not None:
                failed += 1
                log.opt(exception=worker_error).error(
                    f"Worker for {job_queue_path} raised before writing results"
                )
                continue

            process_result = completed.result()
            if process_result.returncode != 0:
                log.error(
                    f"Worker for {job_queue_path} exited with code "
                    f"{process_result.returncode}; see {worker_log_path}"
                )

            try:
                with open(job_queue_path.replace(".pkl", ".results.pkl"), "rb") as f:
                    result = pickle.load(f)
            except Exception as e:
                failed += 1
                log.error(f"Could not read results for {job_queue_path}: {e}")
                log.error(f"Worker subprocess output, if any, is in {worker_log_path}")
                continue

            results.extend(result)
            for r in result:
                if r.status == "success":
                    successful += 1
                    log.info(
                        f"SUCCESS ({r.protein}, {r.model_name}, {r.method}, {r.scaler} "
                        f"{r.runtime_seconds:.1f}s): {r.log_path}"
                    )
                else:
                    failed += 1
                    log.error(
                        f"FAILED ({r.protein}, {r.model_name}, {r.method}, {r.scaler} "
                        f"exit={r.exit_code}): {r.log_path}"
                    )
    return results


def run_guidance_queue_script(
    job_queue_path: str,
    model: str,
    worker_num: int,
    gpus: list[str],
) -> subprocess.CompletedProcess[Any]:
    """Run one pickled guidance job queue in the model's pixi environment.

    Parameters
    ----------
    job_queue_path : str
        Pickled queue of guidance jobs assigned to this worker.
    model : str
        Structure predictor name used to select the pixi environment.
    worker_num : int
        Zero-based worker index. This determines the local CUDA ordinal.
    gpus : list of str
        Selected GPU entries. CUDA remaps entries such as ``4,5`` to local
        process indices ``0,1``.

    Returns
    -------
    subprocess.CompletedProcess
        Result from the subprocess that ran the worker queue.
    """
    pixi_env_name = get_pixi_env(model)
    script_path = Path(__file__).parent / "scripts" / "run_guidance_pipeline.py"
    env_python = get_pixi_env_python(pixi_env_name)
    if env_python:
        cmd = [env_python, str(script_path), "--job-queue-path", job_queue_path]
        process_env = get_pixi_env_process_env(env_python)
    else:
        cmd = [
            "pixi",
            "run",
            "-e",
            pixi_env_name,
            "python",
            str(script_path),
            "--job-queue-path",
            job_queue_path,
        ]
        process_env = os.environ.copy()
    local_gpu = worker_num % len(gpus)
    requested_gpu = gpus[local_gpu]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_source = "CUDA_VISIBLE_DEVICES"
    else:
        gpu_source = "GPU detection"
    log.info(
        f"Running worker {worker_num}: {cmd} on local CUDA GPU {local_gpu} "
        f"(selected GPU {requested_gpu} via {gpu_source})"
    )

    with open(worker_log_path_for(job_queue_path), "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=process_env,
        )
    return result


def get_pixi_env_process_env(env_python: str) -> dict[str, str]:
    """Return process environment values for a direct pixi Python executable.

    Parameters
    ----------
    env_python : str
        Python executable under ``.pixi/envs/<env>/bin/python``.

    Returns
    -------
    dict of str to str
        Environment with the env's ``bin`` directory, ``CONDA_PREFIX``, and
        ``CUDA_HOME`` set so compiled extensions can find tools such as
        ``ninja`` and the CUDA toolkit without going through ``pixi run``.
    """
    env_dir = Path(env_python).resolve().parent.parent
    bin_dir = env_dir / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(env_dir)
    env.setdefault("CUDA_HOME", str(env_dir))
    env["PYTHONNOUSERSITE"] = "1"
    return env


def get_pixi_env_python(pixi_env: str) -> str | None:
    """Return a direct Python binary for a preinstalled pixi environment.

    The ACTL ``pixi-with-checkpoints`` image bakes environments under
    ``/app/.pixi``. Using those interpreters directly avoids a runtime
    ``pixi run`` cache refresh on shared storage. Set ``SAMPLEWORKS_FORCE_PIXI=1``
    to force the old behavior.

    Parameters
    ----------
    pixi_env : str
        Pixi environment name such as ``boltz``, ``protenix``, or ``rf3``.

    Returns
    -------
    str or None
        Path to the environment's Python executable, or ``None`` to use pixi.
    """
    if os.environ.get("SAMPLEWORKS_FORCE_PIXI", "").lower() in {"1", "true", "yes"}:
        return None

    env_key = pixi_env.upper().replace("-", "_")
    override = os.environ.get(f"SAMPLEWORKS_{env_key}_PYTHON")
    if override:
        return override

    pixi_project_dir = Path(os.environ.get("SAMPLEWORKS_PIXI_PROJECT_DIR", "/app"))
    candidate = pixi_project_dir / ".pixi" / "envs" / pixi_env / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def main(args: argparse.Namespace):
    """
    Main pipeline for running grid search trials.
    Args:
        args: Command-line arguments.
    """
    gpus = detect_gpus()
    log.info(f"Detected {len(gpus)} GPUs: {gpus}")
    if args.max_parallel != "auto":
        gpus = gpus[: int(args.max_parallel)]
    if not gpus:
        raise ValueError(
            "No CUDA GPUs are visible; unset CUDA_VISIBLE_DEVICES=none or use a GPU pod"
        )

    log_args(args, gpus)

    filtered_jobs, job_statuses = generate_and_filter_jobs(args)

    if len(filtered_jobs) == 0:
        log.info("No jobs to run!")
        return

    config = GridSearchConfig(
        model_name=args.model,
        scalers=args.scalers.split(),
        ensemble_sizes=[int(x) for x in args.ensemble_sizes.split()],
        gradient_weights=[float(x) for x in args.gradient_weights.split()],
        gd_steps=[int(x) for x in args.num_gd_steps.split()],
        method=args.method,
        proteins_file=args.proteins,
        output_dir=args.output_dir,
    )

    start_time = time.time()
    results = run_grid_search(filtered_jobs, gpus, args, job_statuses=job_statuses)

    if not args.dry_run and results:
        save_results(results, config, args.output_dir, time.time() - start_time)

    log.info("=" * 50)
    log.info("Grid search complete")
    log.info("=" * 50)


def generate_jobs(args: argparse.Namespace) -> list[JobConfig]:
    """Expand CLI grid dimensions into concrete per-protein guidance jobs."""
    jobs = []

    proteins = ProteinInput.from_csv(Path(args.proteins))
    model = args.model
    scalers = args.scalers.split()
    ensemble_sizes = [int(x) for x in args.ensemble_sizes.split()]
    gradient_weights = [float(x) for x in args.gradient_weights.split()]
    gd_steps_list = [int(x) for x in args.num_gd_steps.split()]

    for protein in proteins:
        structure = protein.structure
        density = str(protein.density)
        resolution = protein.resolution
        protein_name = protein.name

        method_suffix = f"_{args.method.replace(' ', '_')}" if args.method else ""
        for scaler in scalers:
            if scaler == GuidanceType.FK_STEERING:
                for ens in ensemble_sizes:
                    for gw in gradient_weights:
                        for gd in gd_steps_list:
                            output_dir = os.path.join(
                                args.output_dir,
                                protein_name,
                                f"{model}{method_suffix}",
                                scaler,
                                f"ens{ens}_gw{gw}_gd{gd}",
                            )
                            log_path = os.path.join(output_dir, "run.log")
                            jobs.append(
                                JobConfig(
                                    protein=protein_name,
                                    structure_path=structure,
                                    density_path=density,
                                    resolution=resolution,
                                    model_name=model,
                                    scaler=scaler,
                                    ensemble_size=ens,
                                    gradient_weight=gw,
                                    gd_steps=gd,
                                    method=args.method,
                                    output_dir=output_dir,
                                    log_path=log_path,
                                )
                            )
            else:
                for ens in ensemble_sizes:
                    for gw in gradient_weights:
                        output_dir = os.path.join(
                            args.output_dir,
                            protein_name,
                            f"{model}{method_suffix}",
                            scaler,
                            f"ens{ens}_gw{gw}",
                        )
                        log_path = os.path.join(output_dir, "run.log")
                        jobs.append(
                            JobConfig(
                                protein=protein_name,
                                structure_path=structure,
                                density_path=density,
                                resolution=resolution,
                                model_name=model,
                                scaler=scaler,
                                ensemble_size=ens,
                                gradient_weight=gw,
                                gd_steps=1,
                                method=args.method,
                                output_dir=output_dir,
                                log_path=log_path,
                            )
                        )

    return jobs


print_lock = Lock()


def save_results(
    results: list[JobResult],
    config: GridSearchConfig,
    output_dir: str,
    total_time: float,
):
    """Merge the latest job results into ``results.json`` under ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    existing_runs = []
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing_data = json.load(f)
                existing_runs = existing_data.get("runs", [])
            log.info(f"Loaded {len(existing_runs)} existing results")
        except Exception as e:
            log.warning(f"Could not load existing results: {e}")

    new_run_keys = {
        (
            r.protein,
            r.model_name,
            r.method,
            r.scaler,
            r.ensemble_size,
            r.gradient_weight,
            r.gd_steps,
        )
        for r in results
    }

    merged_runs = [asdict(r) for r in results]
    for existing_run in existing_runs:
        normalized_run = existing_run.copy()
        if "model" in normalized_run:
            normalized_run.setdefault("model_name", normalized_run.pop("model"))
        key = (
            normalized_run.get("protein"),
            normalized_run.get("model_name"),
            normalized_run.get("method"),
            normalized_run.get("scaler"),
            normalized_run.get("ensemble_size"),
            normalized_run.get("gradient_weight"),
            normalized_run.get("gd_steps"),
        )
        if key not in new_run_keys:
            merged_runs.append(normalized_run)

    output = {
        "config": asdict(config),
        "runs": merged_runs,
        "summary": {
            "total": len(merged_runs),
            "successful": sum(1 for r in merged_runs if r.get("status") == "success"),
            "failed": sum(1 for r in merged_runs if r.get("status") == "failed"),
            "total_runtime_seconds": round(total_time, 2),
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Results saved to {results_path} ({len(merged_runs)} total runs)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one model-specific grid search."""
    parser = argparse.ArgumentParser(
        description="Run grid search across scalers, and parameters for a single "
        "protein structure predictor model."
    )
    # Experiment level arguments
    parser.add_argument(
        "--proteins", required=True, help="CSV file with columns: structure,density,resolution,name"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        default="boltz2",
        choices=["boltz1", "boltz2", "protenix", "rf3", "protpardelle"],
        help="The protein structure predictor model to use",
    )
    parser.add_argument(
        "--model-checkpoint",
        default="",
        help="Override the default checkpoint path for the selected model",
    )
    parser.add_argument(
        "--method",
        default="X-RAY DIFFRACTION",
        choices=["X-RAY DIFFRACTION", "MD"],
        help="Method for Boltz2 ('X-RAY DIFFRACTION', 'MD')",
    )
    parser.add_argument(
        "--recycling-steps",
        type=int,
        default=None,
        help="Number of recycling steps for model inference. If not specified, "
        "uses model default, which can be found in each model's wrapper.py file",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=200,
        help="Number of diffusion steps for model inference. If not specified, "
        "uses model default, which can be found in each model's wrapper.py file",
    )

    # Trajectory scaling arguments
    parser.add_argument(
        "--scalers", default="pure_guidance fk_steering", help="Space-separated scalers"
    )
    parser.add_argument(
        "--ensemble-sizes", default="1 2 4 8", help="Space-separated ensemble sizes"
    )
    parser.add_argument(
        "--gradient-weights",
        default="0.01 0.1 0.2",
        help="Space-separated gradient weights/step sizes",
    )
    parser.add_argument(
        "--partial-diffusion-step", type=int, default=0, help="Partial diffusion step"
    )
    parser.add_argument(
        "--guidance-start",
        type=int,
        default=-1,
        help="Guidance start step for model inference, defaults to -1, meaning start immediately",
    )

    parser.add_argument(
        "--num-gd-steps",
        default="20",
        help="Space-separated GD steps (FK steering only)",
    )
    parser.add_argument("--num-particles", type=int, default=3, help="FK steering: num particles")
    parser.add_argument("--fk-lambda", type=float, default=0.5, help="FK steering: lambda")
    parser.add_argument(
        "--fk-resampling-interval",
        type=int,
        default=1,
        help="FK steering: resampling interval",
    )

    # Step Scaler arguments
    parser.add_argument(
        "--step-scaler-type",
        type=str,
        default="noisespace",
        choices=["dataspace", "noisespace", "none"],
        help="Type of step scaler to use (pure guidance only)",
    )
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
    parser.add_argument("--augmentation", action="store_true", help="Enable augmentation")
    parser.add_argument("--align-to-input", action="store_true", help="Align to input structure")

    # Latent optimization (IT-opt) arguments, used when --scalers includes latent_opt.
    # Registered from the shipped adder rather than restated here so that flag names, types and
    # defaults stay identical to the sampleworks-guidance CLI.
    add_latent_opt_args(parser)

    # RF3-specific arguments
    parser.add_argument(
        "--disable-chiral-features",
        action="store_true",
        help="Disable RF3 chiral gradient feature during guidance",
    )
    parser.add_argument(
        "--track-chiral-features",
        action="store_true",
        help="Log RF3 chiral gradient magnitudes at each denoising step",
    )

    # Reward/Loss function arguments
    parser.add_argument("--loss-order", type=int, default=2, help="L1 (1) or L2 (2) loss")

    # Output arguments
    parser.add_argument("--output-dir", default="./grid_search_results", help="Output directory")

    # Arguments for choosing what to run and what hardware to use.
    parser.add_argument(
        "--max-parallel",
        default="auto",
        help="Max parallel jobs (default: auto = number of GPUs)",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        choices=(1, 2),
        help="Concurrent jobs per GPU (default: 1). Each holds its own copy of the model weights, "
        "so 2 needs roughly twice the VRAM; 2 is the tested ceiling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-run all jobs, including successful ones (overrides default)",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Run only failed jobs, skip un-run and successful jobs",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Run only un-run jobs, skip failed and successful jobs",
    )

    return parser.parse_args()


def log_args(args: argparse.Namespace, gpus: list[str]):
    """Log the resolved grid-search configuration before jobs are generated."""
    log.info("=" * 50)
    log.info("Starting grid search")
    log.info(f"Model: {args.model}")
    if args.model == "boltz2":
        log.info(f"Boltz2 method: {args.method}")
    if args.model == "rf3":
        log.info(f"Disable chiral features: {args.disable_chiral_features}")
        log.info(f"Track chiral features: {args.track_chiral_features}")
    log.info(f"Scalers: {args.scalers}")
    log.info(f"Ensemble sizes: {args.ensemble_sizes}")
    log.info(f"Gradient weights: {args.gradient_weights}")
    log.info(f"GD steps: {args.num_gd_steps}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"GPUs: {gpus}")
    log.info(f"Dry run: {args.dry_run}")
    log.info("=" * 50)


# TODO make job statuses a proper class
# TODO: there are many constants here like "not_run" that should be defined in only one place.
def generate_and_filter_jobs(args: argparse.Namespace) -> tuple[list[JobConfig], dict[Any, Any]]:
    """Generate jobs and filter them according to prior status and rerun flags."""
    jobs = generate_jobs(args)
    log.info(f"Generated {len(jobs)} total jobs")

    log.info("Checking job statuses...")
    job_statuses = {}
    for job in jobs:
        status = get_job_status(job)
        job_statuses[id(job)] = status

    successful_count = sum(1 for s in job_statuses.values() if s == "success")
    failed_count = sum(1 for s in job_statuses.values() if s == "failed")
    not_run_count = sum(1 for s in job_statuses.values() if s == "not_run")

    log.info(
        f"Status: {successful_count} successful, {failed_count} failed, {not_run_count} not run"
    )

    if args.force_all:
        filtered_jobs = jobs
        log.info("Running all jobs (--force-all)")
    elif args.only_failed:
        filtered_jobs = [job for job in jobs if job_statuses[id(job)] == "failed"]
        log.info(f"Running only failed jobs (--only-failed): {len(filtered_jobs)} jobs")
    elif args.only_missing:
        filtered_jobs = [job for job in jobs if job_statuses[id(job)] == "not_run"]
        log.info(f"Running only un-run jobs (--only-missing): {len(filtered_jobs)} jobs")
    else:
        filtered_jobs = [job for job in jobs if job_statuses[id(job)] in ("failed", "not_run")]
        log.info(f"Running failed and un-run jobs (default): {len(filtered_jobs)} jobs")
    return filtered_jobs, job_statuses


if __name__ == "__main__":
    args = parse_args()
    main(args)
