#!/bin/bash
# Run a command (RD-Agent LLM worker) under a DENY-DEFAULT macOS sandbox-exec profile.
# Usage: sandbox_run.sh <writable_dir> [VAR=VALUE]... -- <cmd...>     (cmd[0] must be an absolute interpreter path)
# Boundary (NOT a container/VM; does not defeat a kernel-level attacker):
#   - env scrubbed (env -i) BEFORE entering the sandbox; HOME/TMPDIR point inside <writable_dir>
#   - process-exec: ONLY cmd[0] (+ its real path and Python.framework version dir). No sh, open, osascript, curl, git...
#   - mach-lookup: only what system.sb grants (no LaunchServices); writes to any *.app path denied
#   - file reads: /Users, /Volumes, /private/tmp, /private/var/folders denied except <writable_dir> and the lab root
#     (envs, adapters); secret dirs and /Users/newman/quantalpha/data and tradesea denied on top
#   - file writes: only <writable_dir> and /dev/null; never any *.app bundle
#   - network: loopback IP only (local Ollama). No outbound internet, no DNS.
#   - signals: same-sandbox only
# Replaced an `(allow default)` profile that an adversarial review showed could be escaped via LaunchServices (`open` of an .app).
set -euo pipefail
W="$(cd "$1" && pwd -P)"; shift
ENVV=(); while [ "$1" != "--" ]; do
  case "$1" in *=*) ENVV+=("$1");; *) echo "expected VAR=VALUE or --, got: $1" >&2; exit 2;; esac; shift
done; shift
CMD0="$1"
case "$CMD0" in /*) ;; *) echo "cmd[0] must be an absolute path" >&2; exit 2;; esac
LAB="$(cd "$(dirname "$0")/.." && pwd -P)"
UH="/Users/$(id -un)"
for v in "$W" "$CMD0" "$LAB"; do case "$v" in *\"*|*\\*) echo "bad path" >&2; exit 2;; esac; done
case "$CMD0" in "$W"/*) echo "interpreter must not be inside the writable dir" >&2; exit 2;; esac

REAL0="$(/usr/bin/python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$CMD0" 2>/dev/null || echo "$CMD0")"
exec_rules=" (literal \"$CMD0\") (literal \"$REAL0\")"
d="$REAL0"
while [ "$d" != "/" ] && [ -n "$d" ]; do                       # framework builds re-exec Python.app inside Versions/3.x
  d="$(dirname "$d")"
  case "$(basename "$(dirname "$d")")" in Versions) exec_rules="$exec_rules (subpath \"$d\")"; break;; esac
done
meta=""; m="$W"; while [ "$m" != "/" ]; do meta="$meta (literal \"$m\")"; m="$(dirname "$m")"; done
m="$LAB"; while [ "$m" != "/" ]; do meta="$meta (literal \"$m\")"; m="$(dirname "$m")"; done

P="$(mktemp "${TMPDIR:-/tmp}/qal_sandbox.XXXXXX")"            # profile lives OUTSIDE the writable dir
trap 'rm -f "$P"' EXIT
cat > "$P" <<SB
(version 1)
(deny default)
(import "system.sb")
(allow process-fork)
(allow process-exec*$exec_rules)
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow ipc-posix-sem)
(allow file-read*)
(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders") (subpath "/private/var/root") (subpath "/Library/Keychains") (subpath "/private/var/db/dslocal"))
(allow file-read* (subpath "$W") (subpath "$LAB"))
(allow file-read-metadata$meta)
(deny file-read* (subpath "$UH/.ssh") (subpath "$UH/.aws") (subpath "$UH/.claude") (subpath "$UH/.config") (subpath "$UH/.gnupg") (subpath "$UH/.docker") (subpath "$UH/Library") (literal "$UH/.netrc") (subpath "/Users/newman/quantalpha/data") (subpath "/Users/newman/quantalpha/tradesea"))
(allow file-write* (subpath "$W") (literal "/dev/null"))
(deny file-write* (regex #"\.app(/|\$)"))
(allow network-outbound (remote ip "localhost:*"))
(allow network-inbound (local ip "localhost:*"))
(allow network-bind (local ip "localhost:*"))
SB
mkdir -p "$W/.home" "$W/.tmp"
# no exec(): the trap must remove the profile; sandbox-exec is run as a child
/usr/bin/env -i PATH="/usr/bin:/bin" HOME="$W/.home" TMPDIR="$W/.tmp" PYTHONDONTWRITEBYTECODE=1 ${ENVV[@]+"${ENVV[@]}"} \
  /usr/bin/sandbox-exec -f "$P" "$@"
