# Enter ext by default. Set EXT_SHELL=0 before shell start to opt out.
[ "${EXT_SHELL:-1}" != "0" ] || return 0 2>/dev/null || exit 0

case "$-" in
  *i*) ;;
  *) return 0 2>/dev/null || exit 0 ;;
esac

[ -t 0 ] || return 0 2>/dev/null || exit 0
[ -t 1 ] || return 0 2>/dev/null || exit 0
[ -z "${SAMPLEWORKS_EXT_SHELL_ATTEMPTED:-}" ] || return 0 2>/dev/null || exit 0
[ -z "${EXT_SHELL_ACTIVE:-}" ] || return 0 2>/dev/null || exit 0
[ -z "${BASH_EXECUTION_STRING:-}" ] || return 0 2>/dev/null || exit 0
[ -z "${ZSH_EXECUTION_STRING:-}" ] || return 0 2>/dev/null || exit 0
# Not just "is ext on PATH" but "does ext actually run" - past this point the
# shell is handed over with exec, so a broken binary would end the session.
# Bounded because this runs synchronously during shell startup: an ext that
# hangs must not hold the prompt hostage. timeout is coreutils and asserted
# at build time in Dockerfile.astera; were it ever missing, this line just
# fails and the shell stays plain, which is the safe direction.
timeout 5 ext --help >/dev/null 2>&1 || return 0 2>/dev/null || exit 0

__sampleworks_ext_config_template="/usr/local/share/sampleworks/astera/ext-config.toml"
__sampleworks_ext_data_home="${XDG_DATA_HOME:-${HOME:-/home/dev}/.local/share}"
__sampleworks_ext_config_dir="${__sampleworks_ext_data_home}/ext"
__sampleworks_ext_config="${__sampleworks_ext_config_dir}/config.toml"
mkdir -p "${__sampleworks_ext_config_dir}" 2>/dev/null || true
if [ ! -e "${__sampleworks_ext_config}" ] && [ -r "${__sampleworks_ext_config_template}" ]; then
  cp "${__sampleworks_ext_config_template}" "${__sampleworks_ext_config}" 2>/dev/null || true
fi
unset __sampleworks_ext_config_template __sampleworks_ext_data_home
unset __sampleworks_ext_config_dir __sampleworks_ext_config

__sampleworks_ext_inner_shell=""
if [ -n "${BASH_VERSION:-}" ]; then
  __sampleworks_ext_inner_shell="bash"
elif [ -n "${ZSH_VERSION:-}" ]; then
  __sampleworks_ext_inner_shell="zsh"
fi

if [ -n "${__sampleworks_ext_inner_shell}" ]; then
  export SAMPLEWORKS_EXT_SHELL_ATTEMPTED=1
  exec ext shell -shell "${__sampleworks_ext_inner_shell}"
fi
unset __sampleworks_ext_inner_shell
