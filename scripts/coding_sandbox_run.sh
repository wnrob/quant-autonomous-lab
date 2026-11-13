#!/bin/bash
# Stricter sandbox for the coding-agent workflow (engine/coding). Runs UNTRUSTED (patched) code.
# Usage: coding_sandbox_run.sh <writable_dir> <profile_out> <ro_paths colon-sep> <exec_paths colon-sep> [VAR=VALUE]... -- <cmd...>
#        coding_sandbox_run.sh --print-profile <writable_dir> <ro_paths> <exec_paths>   (prints the profile, runs nothing)
# Boundary (macOS sandbox-exec, DENY-DEFAULT profile; NOT a container/VM, it does not defeat a kernel-level attacker):
#   - env scrubbed (env -i) before entering the sandbox; HOME/TMPDIR point inside <writable_dir>
#   - process-exec: ONLY the listed <exec_paths> (the interpreter). No /bin/sh, /usr/bin/open, osascript, curl, git, ...
#   - mach-lookup: only what system.sb grants (no LaunchServices, no securityd, no pasteboard, no launchd job submission)
#   - file reads: /Users, /Volumes, /private/tmp, /private/var/folders denied except <writable_dir> and the read-only paths;
#     secret dirs and /Users/newman/quantalpha/data are denied even if listed
#   - file writes: only inside <writable_dir> (and /dev/null); never <writable_dir>/.git, never any *.app bundle
#   - network: only loopback IP (no outbound internet, no DNS, no unix sockets)
#   - signals: only to processes in the same sandbox
# <profile_out> must live OUTSIDE <writable_dir> so the sandboxed process cannot rewrite the evidence.
set -euo pipefail
PRINT=0; if [ "${1:-}" = "--print-profile" ]; then PRINT=1; shift; fi
W="$(cd "$1" && pwd -P)"; shift
if [ "$PRINT" = 1 ]; then P=""; else P="$1"; shift; fi
RO_RAW="$1"; shift
EXEC_RAW="$1"; shift
UH="/Users/$(id -un)"

ro_rules=""; meta_rules=""
add_meta() {                                   # stat() on ancestors only (needed for getcwd/realpath); no directory listing
  local d="$1"
  while [ "$d" != "/" ] && [ -n "$d" ]; do meta_rules="$meta_rules (literal \"$d\")"; d="$(dirname "$d")"; done
}
add_meta "$W"
IFS=':' read -r -a RO <<< "$RO_RAW"
for r in ${RO[@]+"${RO[@]}"}; do
  [ -z "$r" ] && continue
  case "$r" in /*) ;; *) echo "ro path must be absolute: $r" >&2; exit 2;; esac
  case "$r" in *\"*|*\\*) echo "bad ro path" >&2; exit 2;; esac
  if [ -d "$r" ]; then ro_rules="$ro_rules (subpath \"$r\")"; else ro_rules="$ro_rules (literal \"$r\")"; fi
  add_meta "$r"
done
case "$W" in *\"*|*\\*) echo "bad writable dir" >&2; exit 2;; esac
exec_rules=""
IFS=':' read -r -a EX <<< "$EXEC_RAW"
for r in ${EX[@]+"${EX[@]}"}; do
  [ -z "$r" ] && continue
  case "$r" in /*) ;; *) echo "exec path must be absolute: $r" >&2; exit 2;; esac
  case "$r" in *\"*|*\\*) echo "bad exec path" >&2; exit 2;; esac
  case "$r" in "$W"|"$W"/*) echo "exec path must not be inside the writable dir" >&2; exit 2;; esac
  if [ -d "$r" ]; then exec_rules="$exec_rules (subpath \"$r\")"; else exec_rules="$exec_rules (literal \"$r\")"; fi
done
[ -z "$exec_rules" ] && { echo "exec allowlist is empty" >&2; exit 2; }

PROFILE="(version 1)
(deny default)
(import \"system.sb\")
(allow process-fork)
(allow process-exec*$exec_rules)
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow ipc-posix-sem)
(allow file-read*)
(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") (subpath \"/private/tmp\") (subpath \"/private/var/folders\") (subpath \"/private/var/root\") (subpath \"/Library/Keychains\") (subpath \"/private/var/db/dslocal\"))
(allow file-read* (subpath \"$W\")$ro_rules)
(allow file-read-metadata$meta_rules)
(deny file-read* (subpath \"$UH/.ssh\") (subpath \"$UH/.aws\") (subpath \"$UH/.claude\") (subpath \"$UH/.config\") (subpath \"$UH/.gnupg\") (subpath \"$UH/.docker\") (subpath \"$UH/Library\") (literal \"$UH/.netrc\") (subpath \"/Users/newman/quantalpha/data\") (subpath \"/Users/newman/quantalpha/tradesea\"))
(allow file-write* (subpath \"$W\") (literal \"/dev/null\"))
(deny file-write* (subpath \"$W/.git\") (regex #\"\\.app(/|\$)\"))
(allow network-outbound (remote ip \"localhost:*\"))
(allow network-inbound (local ip \"localhost:*\"))
(allow network-bind (local ip \"localhost:*\"))"

if [ "$PRINT" = 1 ]; then printf '%s\n' "$PROFILE"; exit 0; fi
case "$P" in "$W"/*) echo "profile_out must be outside the writable dir" >&2; exit 2;; esac
ENVV=(); while [ "$1" != "--" ]; do
  case "$1" in *=*) ENVV+=("$1");; *) echo "expected VAR=VALUE or --, got: $1" >&2; exit 2;; esac; shift
done; shift
printf '%s\n' "$PROFILE" > "$P"
mkdir -p "$W/.coding_home" "$W/.coding_tmp"
# env is scrubbed BEFORE entering the sandbox, so /usr/bin/env itself never needs to be executable inside it
exec /usr/bin/env -i PATH="/usr/bin:/bin" HOME="$W/.coding_home" TMPDIR="$W/.coding_tmp" \
  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg ${ENVV[@]+"${ENVV[@]}"} /usr/bin/sandbox-exec -f "$P" "$@"
