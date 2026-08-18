#!/usr/bin/env bash
# Install lunus.sf -- differentiable structure factors and diffuse scattering --
# into the active environment, from its own repository.
#
# lunus is developed at github.com/lanl/lunus. It is deliberately not vendored
# into this repo and not synced to pods (see .actlignore), so each environment
# that needs the diffuse guidance target installs it once with this script.
#
# lunus is NOT declared in pyproject.toml, and that is not an oversight.
#
# Declaring it means re-locking, and this workspace cannot be re-locked on any
# single machine. Every environment carries `sampleworks` as an editable path
# dependency, and a source dependency must be BUILT to yield its metadata --
# per platform. macOS cannot build the linux-64 environments; the Linux runner
# cannot build osx-arm64 for boltz-osx. `pixi lock` is all-or-nothing, with no
# --environment or --platform selector, so neither machine can produce a
# complete lock. This is a workspace-wide constraint that predates lunus and
# blocks any dependency addition.
#
# Until that is resolved -- by dropping a platform, or by lunus publishing a
# wheel so no build is needed anywhere -- installation is this script.
# See DIFFUSE_SCATTERING_PLAN.md.
#
# Usage (from the repo root, inside the environment you want it in):
#   pixi run -e boltz bash scripts/install_lunus.sh
#   LUNUS_REF=main pixi run -e boltz bash scripts/install_lunus.sh
set -euo pipefail

LUNUS_REPO="${LUNUS_REPO:-https://github.com/lanl/lunus.git}"
# The differentiable engine lives on the `sf` branch; switch this to main once
# that branch merges.
LUNUS_REF="${LUNUS_REF:-sf}"

echo "Installing lunus[sf] from ${LUNUS_REPO}@${LUNUS_REF}"

# The `sf` extra requires torch and gemmi. Both are already supplied by the pixi
# environment, and lunus leaves them unpinned, so pip treats them as satisfied
# and will not pull a second copy of torch over the conda-provided one.
# --upgrade forces a re-fetch, so re-running picks up new commits on a branch ref.
python -m pip install --upgrade "lunus[sf] @ git+${LUNUS_REPO}@${LUNUS_REF}"

# Verify rather than trust the exit status. lunus.sf resolves its public API
# lazily (PEP 562), so `import lunus.sf` alone succeeds even when the torch-side
# modules are broken or absent -- name the symbols the diffuse reward needs.
python - <<'PY'
from importlib.metadata import version

from lunus.sf import mean_and_diffuse, structure_factors_batch

print(f"lunus {version('lunus')} OK")
print(f"  structure_factors_batch <- {structure_factors_batch.__module__}")
print(f"  mean_and_diffuse        <- {mean_and_diffuse.__module__}")
PY
