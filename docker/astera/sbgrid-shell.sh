# shellcheck shell=sh
# Put the SBGrid collection on $PATH when it is mounted.
#
# SBGrid is a site installation on shared storage, which actl mounts read-only
# at /programs for every diffuse workspace. It is not part of this image, so
# this script must be a no-op whenever the mount is absent (any non-diffuse
# workspace). Every step is guarded accordingly: without the mount, /programs
# is an ordinary empty directory.
#
# Sourced from /etc/profile.d, /etc/bash.bashrc and /etc/zsh/zshrc, so it has
# to stay POSIX and must not rely on word splitting: zsh does not split
# unquoted parameters, which rules out the usual `IFS=: ; for d in $PATH` loop.
#
# Set SBGRID_NO_AUTOINIT=1 before shell start to opt out.

# sbgrid.shrc is written in bash/zsh syntax and says so itself ("Unsupported
# Shell - supported shells are bash and zsh"). It never gets the chance to say
# it under dash: the `[[ ]]` in its first lines is a parse error there, which
# takes down the shell that sourced it -- `|| true` cannot catch that, and
# /bin/sh is dash on this image, so `sh -l` died outright. Only offer the
# collection to the two shells that can actually read it.
if [ -z "${SBGRID_NO_AUTOINIT:-}" ] && [ -z "${SBGRID_INITIALISED:-}" ] \
   && [ -r /programs/sbgrid.shrc ] \
   && { [ -n "${BASH_VERSION:-}" ] || [ -n "${ZSH_VERSION:-}" ]; }; then
    SBGRID_INITIALISED=1
    export SBGRID_INITIALISED

    _sbgrid_path_before="${PATH}"

    # sbgrid.shrc runs commands that return non-zero benignly, so it aborts
    # under `set -e` and would take the whole shell with it. It is also chatty
    # on first run and writes into $HOME. Neither should be able to break shell
    # startup, hence the redirect and the `|| true`.
    . /programs/sbgrid.shrc >/dev/null 2>&1 || true

    # sbgrid.shrc PREPENDS /programs/x86_64-linux/system/sbgrid_bin, a dispatch
    # directory holding an entry for every one of the ~13.6k binaries in the
    # collection. Thirty of those names collide with system binaries -- `curl`
    # and the whole perl toolchain -- so the collection silently wins every one
    # of those lookups.
    #
    # That is not survivable here. SBGrid links its binaries against a host OS,
    # not against this image: its `curl` (the `current` symlink still points at
    # 7.50.3, from 2016) needs the NSS libraries libssl3/libnss3/libnspr4,
    # which this image does not ship, so the shadowed `curl` does not merely
    # lag the system one -- it fails to start at all:
    #
    #   curl: error while loading shared libraries: libssl3.so: cannot open
    #   shared object file: No such file or directory
    #
    # which breaks `curl | bash` installers and anything shelling out to curl
    # (`ext update` among them). Installing the missing NSS libraries would
    # only move the problem: the collection's newer curl 8.9.1 wants
    # libssl.so.1.1, absent here too.
    #
    # So move everything sbgrid.shrc prepended to the END of $PATH. Every
    # SBGrid title still resolves -- phenix, sbgrid-installer and friends are
    # not shadowed by anything -- but the image's own binaries now win the
    # thirty collisions, which is the same rule every other actl workspace
    # image follows (see actl-packages.env in the docker-images repo: images
    # are given the SBGrid prerequisites but do not auto-source the collection,
    # precisely because this shadowing is a nasty surprise).
    if [ "${PATH}" != "${_sbgrid_path_before}" ]; then
        _sbgrid_kept=""
        _sbgrid_moved=""
        _sbgrid_rest="${PATH}"
        while [ -n "${_sbgrid_rest}" ]; do
            case "${_sbgrid_rest}" in
                *:*)
                    _sbgrid_head="${_sbgrid_rest%%:*}"
                    _sbgrid_rest="${_sbgrid_rest#*:}"
                    ;;
                *)
                    _sbgrid_head="${_sbgrid_rest}"
                    _sbgrid_rest=""
                    ;;
            esac
            [ -n "${_sbgrid_head}" ] || continue

            # Demote only what this sourcing added, and only under /programs:
            # an entry the user already had stays exactly where it was.
            case "${_sbgrid_head}" in
                /programs|/programs/*)
                    case ":${_sbgrid_path_before}:" in
                        *":${_sbgrid_head}:"*)
                            _sbgrid_kept="${_sbgrid_kept:+${_sbgrid_kept}:}${_sbgrid_head}"
                            ;;
                        *)
                            _sbgrid_moved="${_sbgrid_moved:+${_sbgrid_moved}:}${_sbgrid_head}"
                            ;;
                    esac
                    ;;
                *)
                    _sbgrid_kept="${_sbgrid_kept:+${_sbgrid_kept}:}${_sbgrid_head}"
                    ;;
            esac
        done

        PATH="${_sbgrid_kept:+${_sbgrid_kept}:}${_sbgrid_moved}"
        PATH="${PATH%:}"
        export PATH

        unset _sbgrid_kept _sbgrid_moved _sbgrid_rest _sbgrid_head
    fi

    unset _sbgrid_path_before
fi
