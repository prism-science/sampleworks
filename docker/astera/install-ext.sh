#!/usr/bin/env bash
set -euo pipefail

EXT_VERSION="${EXT_VERSION:-v0.2.0}"
EXT_INSTALL_DIR="${EXT_INSTALL_DIR:-/usr/local/bin}"

curl -fsSL https://extshell.org/install.sh | bash -s -- --version "${EXT_VERSION}" --dir "${EXT_INSTALL_DIR}"
install -d -m 0755 /home/dev/.local/share/ext
install -m 0644 /usr/local/share/sampleworks/astera/ext-config.toml /home/dev/.local/share/ext/config.toml
command -v ext >/dev/null 2>&1
ext --help >/dev/null
