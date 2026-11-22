#!/bin/sh
# Download every checkpoint in manifest.tsv into $1 (default /checkpoints),
# verifying each file's sha256 before it is kept. Any mismatch fails the build.
set -eu

manifest="$(dirname "$0")/manifest.tsv"
out="${1:-/checkpoints}"
tmp="${out}.partial"
mkdir -p "${out}" "${tmp}"

# Files download in parallel: the slowest source (Protenix, a single server in
# Beijing) sets the build time, so the others should not queue behind it.
pids=""
while IFS="$(printf '\t')" read -r dest url sha; do
    case "${dest}" in ''|'#'*) continue ;; esac
    case "${sha}" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
        *) echo "manifest: ${dest} has no sha256 (${sha})" >&2; exit 1 ;;
    esac
    (
        part="${tmp}/$(echo "${dest}" | tr / _)"
        curl -fsSL --retry 8 --retry-all-errors --retry-delay 10 --connect-timeout 30 \
            -o "${part}" "${url}"
        echo "${sha}  ${part}" | sha256sum -c - >/dev/null \
            || { echo "sha256 mismatch: ${dest} from ${url}" >&2; echo "got $(sha256sum "${part}")" >&2; exit 1; }
        case "${dest}" in
            *.tar) tar -xf "${part}" -C "${out}" && rm -f "${part}" ;;
            *) mkdir -p "$(dirname "${out}/${dest}")" && mv "${part}" "${out}/${dest}" ;;
        esac
        echo "ok ${dest}"
    ) &
    pids="${pids} $!"
done < "${manifest}"

status=0
for pid in ${pids}; do wait "${pid}" || status=1; done
rmdir "${tmp}" 2>/dev/null || true
[ "${status}" -eq 0 ] || { echo "one or more checkpoints failed to download or verify" >&2; exit 1; }
ls -la "${out}"
