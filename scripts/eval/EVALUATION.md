# Evaluation of SampleWorks Grid Search Results

# External software requirements
## tortoize
SampleWorks relies on tortoize to compute backbone and sidechain dihedral angle outliers.
`tortoize` is free software from https://github.com/PDB-REDO/tortoize, and it ships in the `analysis`
and `analysis-dev` pixi environments via bioconda, so `pixi run -e analysis` (and the `external_tools`
preset, which runs its jobs in `analysis`) already has it on `PATH`. No manual install is needed.

If you invoke the script from some other environment, install tortoize yourself and make sure it is
on `PATH`: scripts/eval/run_and_process_tortoize.py checks for the `tortoize` executable before
running and raises an error if it is not available.

## phenix
Information about the phenix package can be found at https://phenix-online.org/. Phenix requires a
license which is free to academic users. Others may have to pay a fee. Sampleworks makes use of the
phenix.clashscore command and `run_and_process_phenix_clashscore.py` will check for it before
running, raising an error if it is not available.

# Running the evaluations
## Preparing the output CIF files
As of this writing, Sampleworks outputs CIF files that primarily contain the output atomic
coordinates, and not the additional information that many programs, like `tortoize` and
`phenix.clashscore`, require. Furthermore, many protein structure predictors effectively
renumber residues. Since our metrics are frequently calculated by comparing selections of atoms or
residues, we must align to the original _sequence_ of the protein as well. Future versions of
Sampleworks will handle these issues automatically. For direct script invocation, you should run
the script `scripts/patch_output_cif_files.py`. This will use the original PDB inputs to
reconstruct proper output CIF files that are numbered correctly and have all necessary metadata to
reconstruct the protein structure correctly.

The `run_analysis` presets automate this step for the `analyze_grid_search`, `all`, and `external_tools`
presets: a sequential `patch_outputs` pre-job runs before evaluation jobs and writes
`refined-patched.cif` files. Override `PATCH_INPUT_PDB_PATTERN`, `PATCH_RCSB_PATTERN`,
`PATCH_CIF_PATTERN`, or `PATCH_DEPTH` if your input or output layout differs.

You can run the following command, which assumes:
- your sampleworks output is stored in `/home/ubuntu/grid_search_results`,
- the output is organized by RCSB PDB ID in directories like `/home/ubuntu/grid_search_results/1VME/...`,
  see the `--rcsb-pattern` argument which is a regex to match the RCSB PDB ID
- the input PDB cif files are stored in `/home/ubuntu/grid_search_inputs` as required for running the
  the grid search (see GRID_SEARCH.md)
- the input PDB cif files are stored in `/home/ubuntu/grid_search_inputs` as required for running the
  the grid search (see GRID_SEARCH.md). The files will have paths like, e.g.,
  `/home/ubuntu/grid_search_inputs/1VME/1VME_original.cif`. See also the `--input-pdb-pattern`
  argument, which is a python format string which must use the `pdb_id` variable to refer to the
  RCSB PDB ID.

```shell
pixi run -e analysis python scripts/patch_output_cif_files.py \
  --input-dir /home/ubuntu/grid_search_results \
  --rcsb-pattern 'grid_search_results/(pdb_[A-Za-z0-9]{8}|[0-9][A-Za-z0-9]{3})/...' \
  --cif-pattern 'refined.cif' \
  --grid-search-input-dir /home/ubuntu/grid_search_inputs \
  --input-pdb-pattern '{pdb_id}/{pdb_id}_original.cif'
```

This script searches recursively for all CIF files under the input directory, by default up to 4
levels deep. If you organize the output more deeply, you can specify the depth with the `--depth`
argument. It will output a patched CIF files named `refined-patched.cif` along each original `refined.cif`
file. These `refined-patched.cif` files can be used as input to the remaining evaluation scripts.

## Running the scripts
The evaluation scripts have a common interface defined by the method
`sampleworks.eval.grid_search_eval_utils.parse_eval_args`. The general form of these commands is:

The preferred ACTL entrypoint is `run_analysis`, which reads TOML presets from `analyses/` and uses
the same backend runner as `run_experiments`:

```shell
export GRID_SEARCH_RESULTS_DIR=/home/ubuntu/grid_search_results
export GRID_SEARCH_INPUTS_DIR=/home/ubuntu/grid_search_inputs
export PROTEIN_CONFIGS_CSV=/home/ubuntu/protein_analysis_config.csv

run_analysis --list
run_analysis analyze_grid_search --jobs rscc,lddt,bond_geometry
run_analysis altloc_find
run_analysis altloc_classify
run_analysis analyze_grid_search --jobs rscc --set defaults.PATCH_INPUT_PDB_PATTERN='{pdb_id}/{pdb_id}_original.cif'
```

The `analyze_grid_search`, `all`, and `external_tools` presets run the CIF patching pre-step before these
evaluation scripts. If you run the scripts directly, run `scripts/patch_output_cif_files.py` first
or point `--target-filename` at files that already contain the required metadata. If patched CIFs
already exist, add `--skip-pre-jobs` to rerun the analyses without repeating the patching step.

For direct script invocation, the equivalent command shape is:

```shell
pixi run -e analysis python scripts/eval/<script> \
--grid-search-results-path /home/ubuntu/grid_search_results \
--grid-search-inputs-path /home/ubuntu/grid_search_inputs \
--target-filename 'refined-patched.cif' \
--depth 4 \
--protein-configs-csv /home/ubuntu/protein_analysis_config.csv \
--occupancies 0.0 0.25 0.5 0.75 1.0 \
--n-jobs 16
```
The `--occupancies` argument is a list of occupancy values to evaluate, which should correspond to
what you used in the grid search.

The `--n-jobs` argument is the number of parallel jobs to run; it is not used by all scripts yet but
speeds some up considerably, especially for the tortoize and clashscore scripts.

The `--depth` argument is the maximum directory depth to recurse into when looking for target CIF files.

The `--protein-configs-csv` argument is a CSV file describes what parts of each protein to evaluate.
Examples can be found in `sampleworks/data/`.
The file has the following columns:
- `protein`, the PDB id of the protein to evaluate.
- `selection`, a semicolon separated list of selections, with a (very) limited PyMOL like algebra
or a more complete atomworks-style selection syntax. See examples in the files in
`sampleworks/data/`. Selections are only used by the RSCC and LDDT scripts.
- `structure_pattern`, the filename of the reference PDB file passed to the sampleworks generation script,
  probably through an input configuration CSV file if you used the `run_grid_search.py` script. If a
  different reference structure was used for different occupancies, you can use the variable `occ_str`
  in the pattern, which is replaced with the occupancy values one by one.
- `map_pattern`, Similar to `structure_pattern`, but for the density map files.
- `base_map_dir` The code assumes the maps for each protein are in their own subdirectory of the inputs
directory specified by `grid_search_inputs_path`, e.g., `processed/1VME`.

## Evaluation scripts you can run.
All evaluation scripts are in the `scripts/eval` directory.

### `run_and_process_tortoize.py`
Uses the PDB-REDO tortoize program to compute backbone and sidechain dihedral angle outliers.
It produces two files in the directory where it is run:
- `tortoize_residues.csv`, detailed information about the each residue's backbone (ramachandran) and
sidechain (\chi_1, \chi_2) angle z-scores. See PDB-REDO tortoize documentation for more details.
- `tortoize_protein_stats.csv` Protein-level aggregations of the tortoize results.

### `run_and_process_phenix_clashscore.py`
Uses the phenix.clashscore program to compute the clashscore of the protein. It produces a single
file in the directory where it is run: `clashscore_metrics.csv`, which contains one row per
_model_ in the CIF file. (If there are 8 models in the CIF file, that CIF file will have 8 rows in
the output file.) For each model, we report the clashscore as defined by the phenix.clashscore
program, and the number of clashes.

### `bond_geometry_eval.py`
This script computes bond-length and bond-angle outliers for every bond and angle in every model in
each CIF file. It produces four files:
- `bond_length_outliers.csv`, one row per outlier bond, with the the CIF file, model number, bond length and the
  RDKit upper and lower bounds for that type of bond.
- `bond_angle_outliers.csv`, one row per outlier angle, with the the CIF file and model number, where an angle outlier is
  defined as having a distance between the two non-central atoms greater than the expected distance
  based on the angle type and the lengths of the two bonds. It also reports the RDKit upper and lower
  _distance_ bounds for that type of angle.
- `bond_length_violation_fractions.csv`, one row per model per CIF file, with the fraction of outlier bonds in that model.
- `bond_angle_violation_fractions.csv`, one row per model per CIF file, with the fraction of outlier angles in that model.

### `rscc_grid_search_script.py`
This script computes the real-space (electron density) correlation coefficient (RSCC) for every
selection defined in the file passed to `--protein-configs-csv`, for every occupancy defined by
`--occupancies`. Model electron density is computed as an average over the models in the CIF file;
target electron density comes from the map file defined by the `base_map_dir` column in the
`--protein-configs-csv` file and the occupancy. The code takes values from voxels in the maps
within 2&#x212B; of the selected atoms' centers. It automatically aligns the maps to the original
protein structure, but for now this requires the original PDB file (extracted again from the protein
config file.). An example row is:


The output file, `rscc_metrics.csv`, contains one row per selection per output CIF.

| column               | description                                                                        | value                                                                                                                   |
|----------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| protein              | an RCSB id                                                                         | 6dur                                                                                                                    |
| altloc_occupancies  | a dictionary indicating occupancies used in the experiments                        | {'A': 1.0}                                                                                                              |
| model                | the model used for generation                                                      | boltz2                                                                                                                  |
| method               | (Boltz-2 only)                                                                     | MD                                                                                                                      |
| scaler               | the trajectory scaler                                                              | pure_guidance                                                                                                           |
| ensemble_size        | The number of output models in the `refined_cif_path`                              | 8                                                                                                                       |
| guidance_weight      |                                                                                    | 0.1                                                                                                                     |
| gd_steps             | Deprecated.                                                                        | N/A                                                                                                                     |
| trial_dir            | the output directory used in generation                                            | /data/sampleworks-exp/occ_sweep/grid_search_results/6DUR_1.0occA/boltz2_MD/pure_guidance/ens8_gw0.1                     |
| refined_cif_path     | the exact CIF file containing generated models                                     | /data/sampleworks-exp/occ_sweep/grid_search_results/6DUR_1.0occA/boltz2_MD/pure_guidance/ens8_gw0.1/refined-patched.cif |
| protein_dir_name     | the subdirectory in --grid-search-results-path containing results for this protein | 6DUR_1.0occA                                                                                                            |
| rscc                 | the real-space electron density for the voxels around selected atoms               | 0.16054469603483204                                                                                                     |
| base_map_path        | The path to the reference/target electron density map used for guidance            | /mnt/diffuse-private/raw/sampleworks/initial_dataset_40_occ_sweeps/processed/6DUR/6DUR_uniform_1.00A.ccp4               |
| error                | Any error, if the calculation cannot be completed                                  |                                                                                                                         |
| selection            | The selection string that was used, described above.                               | chain A and resi 13-46                                                                                                  |


### `lddt_evaluation_script.py`
Similarly to the RSCC script, this script computes the LDDT for every selection defined in the file
passed to `--protein-configs-csv`, for every occupancy defined by `--occupancies`.

This script produces a single file, `lddt_metrics.csv`, with each row as described in the table below.

The script attempts to assign selections in each of the models in the CIF file to the altlocs defined
in the input reference structure, using as a
pseudo-distance the LDDT scores computed over the selected atoms in each model.
In the example below, the CIF file is
```shell
/mnt/diffuse-private/raw/sampleworks/initial_dataset_40/grid_search_results/1VME_native_occ/boltz2_MD/pure_guidance/ens8_gw0.1/refined.cif
```
and it has 8 models. The selection is over all atoms that have altlocs in the reference structure (1VME).
This includes a loop movement, and 5/8 models are closer, in this LDDT sense, to altloc A, while 3/8
are closer to altloc B. The script then computes a silhouette score for this assumed clustering,
which is reported as the `avg_silhouette` score. We also report a pseudo-silhouette score, which is
a measure of how well the generated conformers match the reference altlocs. 1.0 is a perfect score,
and 0.0 indicates a poor clustering. In the example provided, the pseudo-silhouette score is 0.0034,
indicating that the generated conformers are not well separated and do not reflect the reference altlocs.

| column                | description                                                                                                       | example value                                                                                                                             |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| protein               | an RCSB id                                                                                                        | 1vme                                                                                                                                      |
| altloc_occupancies    | ta dictionary indicating occupancies used in the experiments                                                      | {'A': 0.5}                                                                                                                                |
| model                 | the model used for structure generation                                                                           | boltz2                                                                                                                                    |
| method                | (Boltz 2 only) the particular Boltz training method                                                               | MD                                                                                                                                        |
| scaler                | the trajectory scaler used                                                                                        | pure_guidance                                                                                                                             |
| ensemble_size         | the number of models in the output cif file                                                                       | 8                                                                                                                                         |
| guidance_weight       |                                                                                                                   | 0.1                                                                                                                                       |
| gd_steps              | Deprecated                                                                                                        |                                                                                                                                           |
| trial_dir             | the directory containing generation results                                                                       | /mnt/diffuse-private/raw/sampleworks/initial_dataset_40/grid_search_results/1VME_native_occ/boltz2_MD/pure_guidance/ens8_gw0.1            |
| refined_cif_path      | the exact CIF file containing generated models                                                                    | /mnt/diffuse-private/raw/sampleworks/initial_dataset_40/grid_search_results/1VME_native_occ/boltz2_MD/pure_guidance/ens8_gw0.1/refined.cif |
| protein_dir_name      | the subdirectory in --grid-search-results-path containing results for this protein                                | 1VME_native_occ                                                                                                                           |
| rscc                  | unused                                                                                                            | N/A                                                                                                                                       |
| base_map_path         | unused                                                                                                            | N/A                                                                                                                                       |
| error                 | any error, if the calculation cannot be completed                                                                 |                                                                                                                                           |
| selection             | the selection string that was used, described above (full string will be available, shortened here)               | chain_id == 'A' and ((res_id == 1) or ... or (res_id >= 316 and res_id <= 324) or (res_id == 326) or (res_id == 373))                     |
| occupancies           | space-separated list of occupancies of the identified clustered states                                            | [0.625, 0.375]                                                                                                                            |
| avg_silhouette        | the average silhouette score for the clustered conformers/altlocs (see above)                                     | 0.16082633177992867                                                                                                                       |
| avg_silhouette_to_ref | the pseudo-silhouette score of the generated structures to the reference altlocs in the original generation input | 0.003392312238469336                                                                                                                      |
