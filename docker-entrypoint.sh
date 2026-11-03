#!/bin/bash
# Sampleworks Docker Entrypoint
#
# Usage:
#   docker run pixi-with-checkpoints -e <pixi_env> <script> [args...]
#   docker run pixi-with-checkpoints -e boltz run_grid_search.py --proteins /data/proteins.csv ...
#   docker run pixi-with-checkpoints bash  # interactive shell
#
# Available pixi environments: boltz, protenix, rf3, analysis
#
# Examples:
#   # Run grid search with RF3
#   docker run --gpus all -v /data:/data pixi-with-checkpoints \
#     -e rf3 run_grid_search.py \
#     --proteins /data/proteins.csv \
#     --models rf3 \
#     --scalers pure_guidance \
#     --ensemble-sizes "1 4" \
#     --gradient-weights "0.1 0.2" \
#     --output-dir /data/results \
#     --use-tweedie \
#     --rf3-checkpoint /data/checkpoints/rf3.ckpt

set -e

show_help() {
    cat << 'EOF'
Sampleworks - Protein structure prediction with diffusion model guidance

USAGE:
    docker run --gpus all --shm-size=16g pixi-with-checkpoints -e <environment> <script> [arguments...]
    docker run pixi-with-checkpoints bash
    docker run pixi-with-checkpoints --help

IMPORTANT:
    Always use --shm-size=16g (or larger) to avoid shared memory errors with DataLoaders.

OPTIONS:
    -e, --env <env>     Pixi environment to use (boltz, protenix, rf3, analysis)
    -h, --help          Show this help message
    bash                Start an interactive shell

ENVIRONMENTS:
    boltz       For boltz1 and boltz2 models
    protenix    For protenix model
    rf3         For RF3 model
    analysis    For scripts/eval analysis jobs

EXAMPLES:
    # Run grid search with RF3 model
    docker run --gpus all --shm-size=16g -v /data:/data pixi-with-checkpoints \
      -e rf3 run_grid_search.py \
      --proteins /data/proteins.csv \
      --models rf3 \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --use-tweedie \
      --gradient-normalization \
      --augmentation \
      --align-to-input \
      --rf3-checkpoint /data/checkpoints/rf3_foundry_01_24_latest.ckpt

    # Run grid search with Boltz1 model
    docker run --gpus all --shm-size=16g -v /data:/data pixi-with-checkpoints \
      -e boltz run_grid_search.py \
      --proteins /data/proteins.csv \
      --models boltz1 \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --use-tweedie \
      --boltz1-checkpoint /data/checkpoints/boltz1_conf.ckpt

    # Run grid search with Boltz2 model
    docker run --gpus all --shm-size=16g -v /data:/data pixi-with-checkpoints \
      -e boltz run_grid_search.py \
      --proteins /data/proteins.csv \
      --models boltz2 \
      --scalers pure_guidance \
      --methods "X-RAY DIFFRACTION" \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --use-tweedie \
      --boltz2-checkpoint /data/checkpoints/boltz2_conf.ckpt

    # Run grid search with Protenix model
    docker run --gpus all --shm-size=16g -v /data:/data pixi-with-checkpoints \
      -e protenix run_grid_search.py \
      --proteins /data/proteins.csv \
      --models protenix \
      --scalers pure_guidance \
      --ensemble-sizes "1 4" \
      --gradient-weights "0.1 0.2" \
      --output-dir /data/results \
      --use-tweedie \
      --protenix-checkpoint /data/checkpoints/protenix_base_default_v0.5.0.pt

    # Interactive shell
    docker run --gpus all --shm-size=16g -it pixi-with-checkpoints bash

    # Run a custom script
    docker run --gpus all --shm-size=16g -v /data:/data pixi-with-checkpoints \
      -e boltz scripts/boltz2_pure_guidance.py \
      --structure /data/structure.cif \
      --density /data/density.ccp4 \
      --resolution 1.8

GRID SEARCH ARGUMENTS (run_grid_search.py):
    Required:
      --proteins FILE             CSV file with columns: structure,density,resolution,name

    Model selection:
      --models MODEL              Model to use (boltz1, boltz2, protenix, rf3)
                                  Note: Only one model per run currently supported

    Guidance configuration:
      --scalers SCALER            Guidance method (pure_guidance, fk_steering)
      --ensemble-sizes "N M..."   Space-separated ensemble sizes (e.g., "1 4")
      --gradient-weights "X Y..." Space-separated gradient weights (e.g., "0.1 0.2")
      --use-tweedie               Use Tweedie's formula for gradient computation
      --gradient-normalization    Enable gradient normalization
      --augmentation              Enable data augmentation
      --align-to-input            Enable alignment to input structure

    Output:
      --output-dir DIR            Output directory for results
      --dry-run                   Print commands without executing

    Job control:
      --force-all                 Re-run all jobs including successful ones
      --only-failed               Run only failed jobs
      --only-missing              Run only un-run jobs
      --max-parallel N            Max parallel jobs (default: auto = number of GPUs)

    Model-specific options:
      --boltz1-checkpoint PATH    Path to Boltz1 checkpoint (default: /checkpoints/boltz1_conf.ckpt - BAKED IN)
      --boltz2-checkpoint PATH    Path to Boltz2 checkpoint (default: /checkpoints/boltz2_conf.ckpt - BAKED IN)
      --protenix-checkpoint PATH  Path to Protenix checkpoint
      --rf3-checkpoint PATH       Path to RF3 checkpoint
      --methods METHOD            Boltz2 sampling method (default: "X-RAY DIFFRACTION")

BAKED-IN CHECKPOINTS:
    The following checkpoints are pre-installed in the image:
      /checkpoints/boltz1_conf.ckpt                   - Boltz1 model (~3.5GB)
      /checkpoints/boltz2_conf.ckpt                   - Boltz2 model (~2.3GB)
      /checkpoints/ccd.pkl                             - Chemical Component Dictionary (~345MB)
      /checkpoints/rf3_foundry_01_24_latest.ckpt       - RF3 model (~2.9GB)
      /checkpoints/protenix_base_default_v0.5.0.pt     - Protenix model (~1.4GB)

    FK steering options:
      --num-gd-steps "N M..."     Space-separated GD steps (FK steering only)
      --num-particles N           Number of particles for FK steering (default: 3)
      --fk-lambda FLOAT           Weighting factor for resampling (default: 0.5)
      --fk-resampling-interval N  How often to apply resampling (default: 1)

    Advanced:
      --partial-diffusion-start F Reverse-time fraction in [0, 1) to start from (default: 0.0)
      --loss-order N              L1 (1) or L2 (2) loss (default: 2)

METADATA HOST PATH REMAPPING:
    By default, job_metadata.json records container-internal paths (e.g.
    /data/inputs/..., /data/results/...). To record host paths instead, pass
    one or more of these environment variables:

    Split mounts (-v $DATA:/data/inputs -v $RESULTS:/data/results):
      SAMPLEWORKS_HOST_INPUT_DIR       Host path mounted at /data/inputs
      SAMPLEWORKS_HOST_RESULTS_DIR    Host path mounted at /data/results

    Single mount (-v /host/path:/data):
      SAMPLEWORKS_HOST_DIR            Host path mounted at /data

    Example (split mounts):
      docker run -v /mnt/data:/data/inputs:ro -v /results:/data/results \
        -e SAMPLEWORKS_HOST_INPUT_DIR=/mnt/data \
        -e SAMPLEWORKS_HOST_RESULTS_DIR=/results \
        pixi-with-checkpoints -e boltz run_grid_search.py ...

PROTEINS CSV FORMAT:
    The --proteins CSV file must have the following columns:
      name        - Protein identifier
      structure   - Path to input structure file (.cif, .pdb)
      density     - Path to density map file (.ccp4, .mrc, .map)
      resolution  - Map resolution in Angstroms

    Example:
      name,structure,density,resolution
      1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
      2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8

ACTL helper commands:
    run_experiments         Run experiments/*.toml presets
    run_analysis            Run analyses/*.toml presets for scripts/eval

For full argument details, run:
    docker run pixi-with-checkpoints -e boltz run_grid_search.py --help
EOF
}

# Handle special cases first
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# Handle scientist workflow helpers and interactive shells
if [ "$1" = "run_experiments" ] || [ "$1" = "run_experiments.sh" ] || [ "$1" = "run_all_models.sh" ] || [ "$1" = "run_analysis" ] || [ "$1" = "run_analysis.sh" ]; then
    exec "$@"
fi

if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    exec "$@"
fi

# Parse -e/--env argument
ENV=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "Error: -e/--env requires an environment name (boltz, protenix, rf3, analysis)"
                exit 1
            fi
            ENV="$2"
            shift 2
            break
            ;;
        *)
            echo "Error: First argument must be -e <environment>, bash, or --help"
            echo ""
            echo "Usage: docker run pixi-with-checkpoints -e <env> <script> [args...]"
            echo "       docker run pixi-with-checkpoints bash"
            echo "       docker run pixi-with-checkpoints --help"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ -z "$ENV" ]]; then
    echo "Error: Environment not specified. Use -e <env> where env is boltz, protenix, rf3, or analysis"
    echo ""
    echo "Usage: docker run pixi-with-checkpoints -e <env> <script> [args...]"
    echo ""
    echo "Examples:"
    echo "  docker run pixi-with-checkpoints -e boltz run_grid_search.py --proteins /data/proteins.csv"
    echo "  docker run pixi-with-checkpoints -e rf3 run_grid_search.py --help"
    echo "  docker run pixi-with-checkpoints bash"
    exit 1
fi

case $ENV in
    boltz|protenix|rf3|analysis)
        ;;
    *)
        echo "Error: Invalid environment '$ENV'. Must be one of: boltz, protenix, rf3, analysis"
        exit 1
        ;;
esac

# Get the script to run
if [[ $# -eq 0 ]]; then
    echo "Error: No script specified"
    echo "Usage: docker run pixi-with-checkpoints -e <env> <script> [args...]"
    exit 1
fi

SCRIPT="$1"
shift

# If script is "python", run python directly
if [[ "$SCRIPT" == "python" ]]; then
    exec pixi run -e "$ENV" python "$@"
fi

# If script ends in .py, run it with python
if [[ "$SCRIPT" == *.py ]]; then
    # Check if it's a bare script name (like run_grid_search.py)
    if [[ ! -f "$SCRIPT" && -f "/app/$SCRIPT" ]]; then
        SCRIPT="/app/$SCRIPT"
    elif [[ ! -f "$SCRIPT" && -f "/app/scripts/$SCRIPT" ]]; then
        SCRIPT="/app/scripts/$SCRIPT"
    fi
    exec pixi run -e "$ENV" python "$SCRIPT" "$@"
fi

# Otherwise, run command directly via pixi
exec pixi run -e "$ENV" "$SCRIPT" "$@"
