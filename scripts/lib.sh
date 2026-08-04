#!/usr/bin/env bash
# Shared helpers for vector-bench shell scripts.
# Sourced, never executed directly.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VB_ROOT
export VB_SOURCES="${VB_ROOT}/sources"
export VB_WORK="${VB_ROOT}/work"
export VB_DATASETS="${VB_ROOT}/datasets"
export VB_RESULTS="${VB_ROOT}/results"
export VB_CONFIG="${VB_ROOT}/config"
export VB_DOCKER="${VB_ROOT}/docker"
export VB_OVERLAY="${VB_ROOT}/overlay"
export VB_BUILDCTX="${VB_WORK}/buildctx"

# Repositories we read from (never write to). Overridable from the environment
# so the framework can run on a machine where they live elsewhere.
export VB_REPO_MARIADB="${VB_REPO_MARIADB:-$(cd "${VB_ROOT}/.." && pwd)/server}"
export VB_REPO_ALISQL="${VB_REPO_ALISQL:-$(cd "${VB_ROOT}/.." && pwd)/AliSQL}"
export VB_REPO_ANNB="${VB_REPO_ANNB:-$(cd "${VB_ROOT}/.." && pwd)/ann-benchmarks}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_vb_ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

if [[ -t 2 ]]; then
  _C_RESET=$'\033[0m'; _C_BLUE=$'\033[34m'; _C_YEL=$'\033[33m'
  _C_RED=$'\033[31m';  _C_GRN=$'\033[32m';  _C_DIM=$'\033[2m'
else
  _C_RESET=''; _C_BLUE=''; _C_YEL=''; _C_RED=''; _C_GRN=''; _C_DIM=''
fi

log()   { printf '%s%s%s %s\n' "$_C_DIM" "$(_vb_ts)" "$_C_RESET" "$*" >&2; }
info()  { printf '%s%s %s==>%s %s\n' "$_C_DIM" "$(_vb_ts)" "$_C_BLUE" "$_C_RESET" "$*" >&2; }
warn()  { printf '%s%s %sWARN%s %s\n' "$_C_DIM" "$(_vb_ts)" "$_C_YEL" "$_C_RESET" "$*" >&2; }
ok()    { printf '%s%s %s ok %s %s\n' "$_C_DIM" "$(_vb_ts)" "$_C_GRN" "$_C_RESET" "$*" >&2; }
die()   { printf '%s%s %sFAIL%s %s\n' "$_C_DIM" "$(_vb_ts)" "$_C_RED" "$_C_RESET" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

need_cmd() {
  local c
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || die "required command not found: $c"
  done
}

need_docker() {
  need_cmd docker
  docker info >/dev/null 2>&1 \
    || die "cannot talk to the Docker daemon (is it running? are you in the docker group?)"
}

# Refuse to run if it would write into one of the read-only vendor repos.
assert_not_vendor_repo() {
  local target
  target="$(cd "$1" 2>/dev/null && pwd || echo "$1")"
  local repo
  for repo in "$VB_REPO_MARIADB" "$VB_REPO_ALISQL" "$VB_REPO_ANNB"; do
    [[ -d "$repo" ]] || continue
    local canon; canon="$(cd "$repo" && pwd)"
    if [[ "$target" == "$canon" || "$target" == "$canon"/* ]]; then
      die "refusing to write inside read-only vendor repo: $canon"
    fi
  done
}

# ---------------------------------------------------------------------------
# YAML (small, dependency-free reader via python3)
# ---------------------------------------------------------------------------

# yq_get <file> <dotted.path> [default]
# Reads a scalar out of a YAML file. Emits the default (or empty) when absent.
yq_get() {
  local file="$1" path="$2" default="${3-}"
  python3 - "$file" "$path" "$default" <<'PY'
import sys, yaml
path_parts = sys.argv[2].split(".") if sys.argv[2] else []
try:
    with open(sys.argv[1]) as fh:
        node = yaml.safe_load(fh) or {}
except FileNotFoundError:
    print(sys.argv[3]); sys.exit(0)
for part in path_parts:
    if isinstance(node, dict) and part in node:
        node = node[part]
    else:
        print(sys.argv[3]); sys.exit(0)
if node is None:
    print(sys.argv[3])
elif isinstance(node, bool):
    print("true" if node else "false")
elif isinstance(node, (list, tuple)):
    print(",".join(str(x) for x in node))
else:
    print(node)
PY
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

# Deterministic short hash of a set of strings — used for config fingerprints.
vb_hash() { printf '%s' "$*" | sha256sum | cut -c1-12; }

# Human-readable byte count.
human_bytes() { numfmt --to=iec-i --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0}B"; }

# Return 0 if the image exists locally.
image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# Image digest (or "unknown" when unavailable, e.g. never pushed).
image_digest() {
  docker image inspect --format '{{index .RepoDigests 0}}' "$1" 2>/dev/null \
    || docker image inspect --format '{{.Id}}' "$1" 2>/dev/null \
    || echo unknown
}
