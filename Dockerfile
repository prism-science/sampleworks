# syntax=docker/dockerfile:1
# Sampleworks - public multi-stage image
#
# The default build produces the public pixi-with-checkpoints image:
# dependencies, source, and model checkpoints, with no Astera/EXT-specific
# tooling.
#
# Build public image locally:
#   docker build --platform linux/amd64 \
#     --build-arg CHECKPOINTS_IMAGE=<checkpoint-image-ref> \
#     -t diffuseproject/pixi-with-checkpoints:local .
#
# Fast context/Dockerfile smoke check without pulling checkpoints or installing
# pixi environments:
#   docker build --target source-check -t sampleworks-source-check .
#
# Private Astera/EXT overlays are built from Dockerfile.astera after this image is
# pushed to the public registry.

# Public default pinned to the Docker Hub manifest list for reproducible builds.
# Astera CI may override this with a digest-pinned Docker Hub cache mirror.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5

# Required checkpoint layer for the full image. The `scratch` default keeps
# source-check builds cheap; CI must override it with a digest-pinned public image.
ARG CHECKPOINTS_IMAGE=scratch

# ============================================================================
# OS base: CUDA + Pixi + common build/runtime dependencies
# ============================================================================
FROM ${BASE_IMAGE} AS os-base

ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/root/.pixi \
    PATH="/root/.pixi/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    TORCH_CUDA_ARCH_LIST="9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Pinned, because everything else about this build is: the base image is a
# digest, the checkpoints are a digest, and CI's setup-pixi is v0.73.0. An
# unpinned `curl | bash` made the build tool the one component that could change
# under us between two builds of the same commit, which is a bad property to
# have while chasing a build that installs nothing (see the guard below). Keep
# in step with `pixi-version` in .github/workflows/ci.yml.
ARG PIXI_VERSION=v0.73.0
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION="${PIXI_VERSION}" bash

WORKDIR /app

# ============================================================================
# Source/context stage: useful for cheap CI smoke checks
# ============================================================================
FROM os-base AS source

# Copy only what the runtime image needs. The package is installed as an editable
# pixi dependency (`sampleworks = { editable = true, path = "." }`), so source
# must remain in the public image.
COPY pyproject.toml pixi.lock LICENSE README.md ./
COPY analyses/ ./analyses/
COPY experiments/ ./experiments/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_grid_search.py ./
COPY run_experiments run_experiments.sh run_all_models.sh ./
COPY docker-entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 0755 \
        /usr/local/bin/entrypoint.sh \
        ./run_experiments \
        ./run_experiments.sh \
        ./run_all_models.sh \
    && cp ./run_experiments ./run_experiments.sh ./run_all_models.sh /usr/local/bin/

FROM source AS source-check
ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]

# ============================================================================
# Checkpoints: copied from a registry image so the build context stays small
# ============================================================================
FROM ${CHECKPOINTS_IMAGE} AS checkpoints

# ============================================================================
# Pixi environments: install all supported model environments in one layer
# ============================================================================
FROM source AS pixi-envs

# Checkpoints (~10 GB) rarely change, so these layers stay cacheable across most
# source edits and dependency-only rebuilds.
#
# Split per-file so a cold pull fetches them as concurrent S3 flows instead of
# one serial ~10 GB stream. Harbor redirects each blob GET to presigned S3, and
# containerd downloads each layer on its own connection (up to
# max_concurrent_downloads), so N layers pull in parallel where one COPY could
# not. The files are near-incompressible weights, so extra layer boundaries add
# essentially no image size. Largest first: the pull finishes with the slowest
# layer, so the biggest checkpoint should start downloading first.
#
# Do NOT add a trailing catch-all `COPY /checkpoints/ /checkpoints/`. Layers are
# deduplicated per layer, not per file, so re-copying files already present
# writes a second full ~10 GB copy of every checkpoint AND reinstates the single
# serial flow this split exists to remove. The guard below covers new
# checkpoints instead, by failing the build rather than shipping them silently.
COPY --from=checkpoints /checkpoints/boltz1_conf.ckpt                /checkpoints/
COPY --from=checkpoints /checkpoints/rf3_foundry_01_24_latest.ckpt   /checkpoints/
COPY --from=checkpoints /checkpoints/boltz2_conf.ckpt                /checkpoints/
COPY --from=checkpoints /checkpoints/protenix_base_default_v0.5.0.pt /checkpoints/
COPY --from=checkpoints /checkpoints/ccd.pkl                         /checkpoints/
COPY --from=checkpoints /checkpoints/mols/                           /checkpoints/mols/

# Bind mount reads the checkpoint stage without copying any bytes into the
# image. Keep this list in sync with the COPY lines above; bumping
# CHECKPOINTS_SOURCE_IMAGE to a digest with new or renamed files fails here.
RUN --mount=type=bind,from=checkpoints,target=/ck \
    unexpected="$(ls -A /ck/checkpoints | grep -vxE 'boltz1_conf\.ckpt|boltz2_conf\.ckpt|ccd\.pkl|rf3_foundry_01_24_latest\.ckpt|protenix_base_default_v0\.5\.0\.pt|mols')"; \
    [ -z "${unexpected}" ] || { \
        echo "Unenumerated checkpoints, add a COPY line above for each: ${unexpected}"; \
        exit 1; \
    }

# IMPORTANT: keep these installs in a single RUN. Splitting them into separate
# Docker layers duplicates shared conda packages (numpy, CUDA libs, etc.) and can
# add tens of GB to the image.
#
# Published images have shipped all five environments as empty shells:
# conda-meta/pixi exists, but there are no package records, bin/, or lib/.
# `pixi install --frozen` reports success against that state without running a
# package transaction. `pixi reinstall` is the supported command that bypasses
# that up-to-date decision and materializes every package again. It also creates
# a missing prefix, so clearing prefixes first covers both stale layers and the
# empty-prefix failure mode seen on diffuse-sh-builder.
#
# `/root/.cache/pixi` is deliberately not cached across builds — it carries
# pixi's own "is this environment current" state, which is the thing that can
# disagree with a prefix restored from a different build. The rattler and uv
# caches stay: they hold downloaded packages and wheels, are what actually make
# rebuilds fast, and were verified not to affect what lands in the layer.
#
# The cache mounts have project-specific ids so they cannot alias mounts from
# unrelated Dockerfiles when a builder retains state. `sharing=locked` prevents
# concurrent steps on the same builder from mutating a package cache together.
# Bump the suffix only to recover from a demonstrably corrupt persistent cache.
#
# The guard prints diagnostics before it exits. The empty-environment install
# has not been reproducible outside this builder: the same manifest, lock and
# pixi version install all five environments correctly on linux/amd64 with both
# a cold and a warm rattler cache. So when it happens here, the build log is the
# only place the cause can come from, and "pixi said installed, nothing is
# there" is not enough to act on.
RUN --mount=type=cache,id=sampleworks-rattler-2,target=/root/.cache/rattler,sharing=locked \
    --mount=type=cache,id=sampleworks-uv-2,target=/root/.cache/uv,sharing=locked \
    pixi --version && \
    pixi info --no-config && \
    rm -rf /app/.pixi/envs && \
    for env in boltz protenix rf3 protpardelle analysis; do \
        echo "=== reinstalling ${env} ==="; \
        pixi reinstall -e "${env}" --frozen --no-config || { \
            echo "FATAL: pixi failed while reinstalling environment '${env}'."; \
            exit 1; \
        }; \
        python="/app/.pixi/envs/${env}/bin/python"; \
        test -x "${python}" || { \
            echo "FATAL: pixi environment '${env}' has no interpreter at /app/.pixi/envs/${env}/bin/python."; \
            echo "       pixi reinstall reported success but materialized no packages."; \
            echo "--- what pixi left behind ---"; \
            ls -la "/app/.pixi/envs/${env}" 2>&1 | head -20; \
            echo "--- prefix bookkeeping (conda-meta) ---"; \
            ls -A "/app/.pixi/envs/${env}/conda-meta" 2>&1 | head -10; \
            head -c 400 "/app/.pixi/envs/${env}/conda-meta/pixi" 2>&1; echo; \
            echo "--- environments pixi thinks exist ---"; \
            ls -A /app/.pixi/envs 2>&1 | head; \
            echo "--- where pixi is installing to ---"; \
            pixi info --no-config 2>&1 | grep -iE "cache dir|environments|manifest|version" | head; \
            exit 1; \
        }; \
        "${python}" --version || { \
            echo "FATAL: environment '${env}' has an unusable Python interpreter."; \
            exit 1; \
        }; \
    done

# A GPU is not required to build the image. Pre-compile CUDA extensions only when
# the builder exposes NVIDIA devices; if present, failures should stop the build.
RUN if [ ! -e /dev/nvidiactl ] && [ ! -e /proc/driver/nvidia/version ]; then \
        echo "CUDA extension pre-compilation skipped (no GPU visible during build)"; \
    else \
        pixi run -e boltz python -c "\
from sampleworks.core.forward_models.xray.real_space_density_deps.ops import dilate_atom_centric; \
print('CUDA extensions compiled successfully')"; \
    fi

# This image carries pixi environments and checkpoints. Runtime source should
# come from ACTL's synced checkout at /home/dev/workspace, not from stale code
# baked into /app during image construction.
RUN rm -rf /app/src /app/scripts /app/experiments /app/analyses \
    /app/run_grid_search.py /app/run_analysis \
    && mkdir -p /home/dev/workspace

COPY --chmod=755 run_experiments run_experiments.sh run_all_models.sh run_analysis run_analysis.sh /usr/local/bin/
RUN printf '\n# ACTL scientist workflow: land in the synced Sampleworks checkout.\nif [[ $- == *i* ]] && [ -z "${SAMPLEWORKS_NO_AUTO_CD:-}" ] && [ -d /home/dev/workspace ]; then\n    cd /home/dev/workspace\nfi\n' \
    | tee -a /root/.bashrc /home/dev/.bashrc >/dev/null

# ============================================================================
# Public runtime: regular Sampleworks image for the public registry
# ============================================================================
FROM pixi-envs AS public

ENV BOLTZ1_CHECKPOINT=/checkpoints/boltz1_conf.ckpt \
    BOLTZ2_CHECKPOINT=/checkpoints/boltz2_conf.ckpt \
    CCD_PATH=/checkpoints/ccd.pkl \
    RF3_CHECKPOINT=/checkpoints/rf3_foundry_01_24_latest.ckpt \
    PROTENIX_CHECKPOINT=/checkpoints/protenix_base_default_v0.5.0.pt \
    HOME=/home/dev \
    XDG_CONFIG_HOME=/home/dev/.config \
    XDG_CACHE_HOME=/home/dev/.cache \
    XDG_DATA_HOME=/home/dev/.local/share \
    SHELL=/bin/bash

RUN mkdir -p /home/dev/.config /home/dev/.cache /home/dev/.local/share /home/dev/workspace

WORKDIR /home/dev

ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]
