# Running the Sampleworks Grid Search
One of the main tasks in the Sampleworks project is to run a hyperparameter search
over parameters like the number of samples in an ensemble and the guidance weight.
This document describes how to run the grid search, how the results are organized,
and how to find and read logs if you need to debug the process.

## Optional: Setting up the docker container
It is often useful to have a docker container with all the dependencies installed.
Our script `run_experiments` for instance uses a docker container to manage all
dependencies. To run that script, you will need to have docker installed. Build
the container with
```shell
docker build --platform linux/amd64 \
  --build-arg CHECKPOINTS_IMAGE=<checkpoint-image-ref> \
  -t diffuseproject/pixi-with-checkpoints:local .
```
which will add an image to your local docker repository called
`diffuseproject/pixi-with-checkpoints:local`. The top of the `Dockerfile` contains
instructions on how to use the container as well. The container entrypoint
(`docker-entrypoint`) is fairly generic and is used to call the `run_grid_search.py`
script described below.

## Running the Grid Search
To run grid search, you can use the `run_grid_search.py` script. It primarily sweeps
supplied values of gradient weights and ensemble sizes. It requires a CSV file with
protein structure, density map, and resolution columns, described below.

```bash
pixi run -e boltz python run_grid_search.py \
    --proteins proteins.csv \
    --models boltz2 \                # options: boltz1, boltz2, protenix, rf3 (make sure env aligns!)
    --methods "X-RAY DIFFRACTION" \  # only useful for Boltz-2, ignored otherwise
    --scalers pure_guidance \        # options: pure_guidance, fk_steering, or both as space-separated list
    --ensemble-sizes "1 4" \
    --gradient-weights "0.1 0.2" \
    --output-dir grid_search_results \
    --gradient-normalization \       # normalize guidance update magnitude to diffusion update magnitude
    --augmentation \                 # apply random rotations and translations at each step (defaults for inference with AF3-like models)
    --align-to-input                 # align to input structure at each step (required for density guidance to work since it is not rotation/translation invariant)
```

**`proteins.csv` format**

Required columns and format. Supported density map formats: `.ccp4`, `.mrc`, `.map` (not MTZ or SF-CIF yet).
```csv
name,structure,density,resolution
1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8
```

**Key arguments:**

| Argument             | Description                                                | Default                     |
|----------------------|------------------------------------------------------------|-----------------------------|
| `--proteins`         | CSV with structure/density/resolution columns              | required                    |
| `--models`           | Model to run. One of `boltz1`, `boltz2`, `protenix`, `rf3` | required                    |
| `--scalers`          | Guidance method(s) to sweep                                | `pure_guidance fk_steering` |
| `--ensemble-sizes`   | Space-separated values, e.g. `"1 4"`                       | `"1 2 4 8"`                 |
| `--gradient-weights` | Space-separated values, e.g. `"0.1 0.2"`                   | `"0.01 0.1 0.2"`            |
| `--methods`          | Boltz-2 sampling method (required for boltz2)              | `X-RAY DIFFRACTION`         |
| `--max-parallel`     | Parallel workers (default: number of GPUs)                 | `auto`                      |
| `--dry-run`          | Print jobs without running them                            | off                         |
| `--force-all`        | Re-run including already-successful jobs                   | off                         |
| `--only-failed`      | Re-run only failed jobs                                    | off                         |
| `--only-missing`     | Run only jobs not yet started                              | off                         |
| `--disable-chiral-features` | Zero out RF3 chiral gradient features               | off                         |
| `--track-chiral-features` | Track RF3 chiral gradient magnitude                   | off                         |

> **Note**: Jobs are skipped if a `refined.cif` file already exists in the output directory.
> Some flags (e.g., `--use-tweedie`, `--gradient-normalization`) are not reflected in the
> directory structure, so changing them alone won't trigger a re-run. Use `--force-all` to
> re-run all jobs regardless. This is under active development and will likely change soon.

## Location and contents of results
Output layout: `grid_search_results/<protein>/<model>[_<method>]/<scaler>/ens<N>_gw<W>/`

This layout is subject to change. Directory names are constructed as follows:

| Name      | Description                                                         |
|-----------|---------------------------------------------------------------------|
| `protein` | a name for the protein, from the proteins.csv file described above. |
| `model`   | the model name, e.g. `boltz2` or `protenix`.                        |
| `method`  | the sampling method, e.g. `X-RAY DIFFRACTION` for Boltz-2.          |
| `scaler`  | the guidance method, e.g. `pure_guidance` or `fk_steering`.         |
| `N`       | the ensemble size.                                                  |
| `W`       | the gradient weight.                                                |

In that directory, you will find the following files generated by Sampleworks
(in addition to model-specific files, e.g. the Boltz imput YAML file):

- refined.cif: the refined structure's mmCIF _atom_site loop.
- job_metadata.json: the full set of parameters used for this ensemble generation trial

> Note: the `refined.cif` file contains _only_ the `_atom_site` loop. This means
> that it is not strictly a valid CIF file and may not be parseable by other programs.
> In particular, we know that it does not work with PDB-REDO scripts and may not work
> with some phenix scripts and programs. We currently have a script `sampleworks/scripts/patch_input_cif_files.py`
> that reconstructs the necessary information to make the CIF file parseable.
> We are tracking the issue https://github.com/prism-science/sampleworks/issues/68 and hope to fix it more generally soon.

## Finding and reading logs
The script `run_grid_search.py` actually creates and runs several subprocesses.
Its output primarily tells you where to find the rest of the output. The top of
the script output will look like this:
```log
2026-03-10 02:01:24.256 | INFO     | __main__:main:241 - Detected 2 GPUs: ['0', '1']
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:525 - ==================================================
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:526 - Starting grid search
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:527 - Models: boltz2
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:528 - Scalers: pure_guidance
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:529 - Ensemble sizes: 8
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:530 - Gradient weights: 0.1 0.2 0.5
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:531 - GD steps: 20
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:532 - Boltz2 methods: X-RAY DIFFRACTION
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:533 - Output directory: /data/results
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:534 - GPUs: ['0', '1']
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:535 - Dry run: False
2026-03-10 02:01:24.256 | INFO     | __main__:log_args:536 - ==================================================
2026-03-10 02:01:24.259 | INFO     | __main__:generate_and_filter_jobs:543 - Generated 600 total jobs
2026-03-10 02:01:24.259 | INFO     | __main__:generate_and_filter_jobs:545 - Checking job statuses...
2026-03-10 02:01:24.270 | INFO     | __main__:generate_and_filter_jobs:555 - Status: 570 successful, 2 failed, 28 not run
2026-03-10 02:01:24.270 | INFO     | __main__:generate_and_filter_jobs:570 - Running failed and un-run jobs (default): 30 jobs
2026-03-10 02:01:24.270 | INFO     | __main__:run_grid_search:147 - Running 30 jobs with 2 parallel workers
2026-03-10 02:01:24.271 | INFO     | __main__:run_grid_search:160 - Pickling worker job queue to /data/results/wjq_140704512577664.pkl
2026-03-10 02:01:24.271 | INFO     | __main__:run_grid_search:160 - Pickling worker job queue to /data/results/wjq_140704503117952.pkl
2026-03-10 02:01:24.271 | INFO     | __main__:run_grid_search:176 - Cleaning existing output directory: /data/results/5IMV_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.1
2026-03-10 02:01:24.271 | INFO     | __main__:run_grid_search:176 - Cleaning existing output directory: /data/results/5MC8_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.1
2026-03-10 02:01:24.463 | INFO     | __mp_main__:run_guidance_queue_script:226 - Running worker 0: ['pixi', 'run', '-e', 'boltz', 'python', '/app/scripts/run_guidance_pipeline.py', '--job-queue-path', '/data/results/wjq_140704512577664.pkl'] on GPU 0
2026-03-10 02:01:24.463 | INFO     | __mp_main__:run_guidance_queue_script:226 - Running worker 1: ['pixi', 'run', '-e', 'boltz', 'python', '/app/scripts/run_guidance_pipeline.py', '--job-queue-path', '/data/results/wjq_140704503117952.pkl'] on GPU 1
```
This output contains basic output about the conditions it is trying, how many GPUs are available,
how many jobs are being run, and where the output is being written.

> If you are running the script with a docker container,
> the output directory is `/data/results` inside the container.
> You would need to look at your docker command to know where that is (or is not)
> linked to your host machine.

As jobs complete, you will see rows like this:
```log
2026-03-10 02:09:10.038 | INFO     | __main__:run_grid_search:204 - SUCCESS (5IMV_1.0occB, boltz2, X-RAY DIFFRACTION, pure_guidance 15.5s): /data/results/5IMV_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.1/run.log
2026-03-10 02:09:10.038 | INFO     | __main__:run_grid_search:204 - SUCCESS (5MC8_1.0occB, boltz2, X-RAY DIFFRACTION, pure_guidance 23.6s): /data/results/5MC8_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.2/run.log
2026-03-10 02:09:10.038 | INFO     | __main__:run_grid_search:204 - SUCCESS (5MHX_1.0occB, boltz2, X-RAY DIFFRACTION, pure_guidance 37.2s): /data/results/5MHX_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.5/run.log
```
which indicate where the log for a specific trial is located. Some debugging information
can be found in those `run.log` files.
However, if there is a serious failure you may not get that far, in which case you will need to look
at the "worker job queue" logs. To find these, find the lines that say "Job failed with exception" in
the output of `run_grid_search.py`, like this one:
```log
2026-03-10 02:03:32.641 | ERROR    | __main__:run_grid_search:216 - Job failed with exception: [Errno 2] No such file or directory: '/data/results/wjq_140011687126848.results.pkl'
```
The `wjq_140011687126848.results.pkl` file is the "worker job queue" results file.
There is a corresponding `wjq_140011687126848.pkl` which contains a `TrialList`
object that specifies the `Trials` that were attempted by a subprocess. The corresponding log file
in this example is `wjq_140011687126848.log`. This file will contain the traceback
of the exception that caused the "worker job queue" to fail. Looking through this file, you can
see what trial is actually being run by looking for lines like this one:
```log
2026-03-10 02:03:23.664 | INFO     | sampleworks.utils.guidance_script_utils:run_guidance_job_queue:589 - Running job 1/15: GuidanceConfig(protein='5I
MV_1.0occB', structure='/data/inputs/processed/5IMV/5IMV_single_001_density_input.cif', density='/data/inputs/occ_sweeps/1.0occB/density_maps/5IMV_1.0
occB_1.00A.ccp4', model_name='boltz2', guidance_type='pure_guidance', log_path='/data/results/5IMV_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.
1/run.log', output_dir='/data/results/5IMV_1.0occB/boltz2_X-RAY_DIFFRACTION/pure_guidance/ens8_gw0.1', partial_diffusion_step=120, loss_order=2, resol
ution=1.0, device='cuda:0', gradient_normalization=True, em=False, guidance_start=-1, augmentation=True, align_to_input=True)
```
which also gives you a path to the `run.log` file for that job. If that job fails, there will be
information both in `run.log` in the trial output directory, as well as below in the worker job
queue log.

If you encounter errors, please share the corresponding worker job queue log `wjq_*.log`
and `run.log` with us, as well as the CIF and map files that were used.
