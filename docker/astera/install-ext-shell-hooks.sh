#!/usr/bin/env bash
set -euo pipefail

profile_script="/etc/profile.d/sampleworks-ext-shell.sh"
profile_comment="# Sampleworks: enter ext by default; EXT_SHELL=0 opts out."
profile_line="[ -r ${profile_script} ] && . ${profile_script}"

touch /root/.bashrc /home/dev/.bashrc

for profile_file in /etc/bash.bashrc /root/.bashrc /home/dev/.bashrc /etc/zsh/zshrc /etc/zsh/zprofile; do
    if [ ! -e "${profile_file}" ]; then
        continue
    fi
    if grep -Fqs "${profile_line}" "${profile_file}"; then
        continue
    fi
    printf '\n%s\n%s\n' "${profile_comment}" "${profile_line}" >> "${profile_file}"
done
