#!/usr/bin/env bash
#
# Build the vector-bench engine images from the pinned sources.
#
# Each engine produces two images:
#   <engine>-runtime  the server alone, for manual use (docs/03-running-manually.md)
#   <engine>-bench    runtime + the ann-benchmarks Python stack
#
# Usage:
#   scripts/build-images.sh [--engine mariadb|mariadb123|alisql|pgvector|all]
#                           [--target runtime|bench|all]
#                           [--march x86-64-v3|native|...]
#                           [--jobs N] [--no-cache] [--pull]
#
# --march is the flag that decides which SIMD path the distance kernels take.
# Every engine gets the SAME value, because otherwise the benchmark compares
# compiler flags rather than implementations. Use `native` only when the build
# host and the benchmark host are the same machine.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ENGINE=all
TARGET=all
MARCH=""
JOBS=0
EXTRA_BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    --march) MARCH="$2"; shift 2 ;;
    --march=*) MARCH="${1#*=}"; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --no-cache) EXTRA_BUILD_ARGS+=(--no-cache); shift ;;
    --pull) EXTRA_BUILD_ARGS+=(--pull); shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_docker
need_cmd python3

# BuildKit is faster and gives better output, but it needs the buildx component,
# which plenty of hosts do not have. The Dockerfiles deliberately avoid
# BuildKit-only syntax so they build either way; pick whichever is available
# rather than failing on a machine that only has the classic builder.
if docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT=1
  info "using BuildKit (buildx present)"
else
  export DOCKER_BUILDKIT=0
  warn "buildx not found; using the classic builder (slower rebuilds, same result)"
fi

# ---------------------------------------------------------------------------

# Read the space-separated cmake flag list out of an engine config.
cmake_flags_for() {
  python3 - "$VB_CONFIG/engines/$1.yml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(" ".join(cfg.get("build", {}).get("cmake_flags", []) or []))
PY
}

build_engine() {
  local engine="$1"
  local cfg="$VB_CONFIG/engines/$engine.yml"
  [[ -f "$cfg" ]] || die "no config for engine '$engine'"

  local ctx="$VB_BUILDCTX/$engine"
  [[ -f "$ctx/source.tar" ]] \
    || die "build context missing for $engine — run: scripts/prepare-sources.sh --engine $engine"

  local tag;      tag="$(yq_get "$cfg" source.tag unknown)"
  local base;     base="$(yq_get "$cfg" image.base ubuntu:24.04)"
  local rt_image; rt_image="$(yq_get "$cfg" image.runtime "vector-bench/${engine}-runtime")"
  local bn_image; bn_image="$(yq_get "$cfg" image.bench   "vector-bench/${engine}-bench")"
  local btype;    btype="$(yq_get "$cfg" build.type RelWithDebInfo)"
  local march;    march="${MARCH:-$(yq_get "$cfg" build.march x86-64-v3)}"
  local cxxextra; cxxextra="$(yq_get "$cfg" build.extra_cxxflags "")"
  local optflags; optflags="$(yq_get "$cfg" build.optflags "")"
  local flags;    flags="$(cmake_flags_for "$engine")"

  local -a bargs=(
    --build-arg "BASE_IMAGE=${base}"
    --build-arg "MARCH=${march}"
    --build-arg "BUILD_TYPE=${btype}"
    --build-arg "CMAKE_FLAGS=${flags}"
    --build-arg "EXTRA_CXXFLAGS=${cxxextra}"
    --build-arg "JOBS=${JOBS}"
  )
  [[ -n "$optflags" ]] && bargs+=(--build-arg "OPTFLAGS=${optflags}")
  # An engine may declare `alias_of` to reuse another engine's Dockerfile and
  # tag build-arg. That is how a second MariaDB version costs one config file
  # instead of a branch in every script.
  local base; base="$(yq_get "$cfg" alias_of "$engine")"
  case "$base" in
    mariadb)  bargs+=(--build-arg "MARIADB_TAG=${tag}") ;;
    alisql)   bargs+=(--build-arg "ALISQL_TAG=${tag}") ;;
    pgvector) bargs+=(--build-arg "PGVECTOR_TAG=${tag}") ;;
    *) die "engine '$engine' has no known build-arg (alias_of=$base)" ;;
  esac

  local -a targets=()
  case "$TARGET" in
    all)     targets=(runtime bench) ;;
    runtime) targets=(runtime) ;;
    bench)   targets=(bench) ;;
    *) die "unknown target: $TARGET" ;;
  esac

  local t img
  for t in "${targets[@]}"; do
    case "$t" in
      runtime) img="$rt_image" ;;
      bench)   img="$bn_image" ;;
    esac
    info "building ${img}:${tag} (target=$t, march=$march, base=$base)"
    local start; start=$(date +%s)
    docker build \
      --target "$t" \
      -t "${img}:${tag}" -t "${img}:latest" \
      -f "$VB_DOCKER/$base/Dockerfile" \
      "${bargs[@]}" "${EXTRA_BUILD_ARGS[@]}" \
      "$ctx" \
      || die "build failed for $engine target $t"
    ok "built ${img}:${tag} in $(( $(date +%s) - start ))s"
  done

  # Record what was actually built, for the run manifest.
  python3 - "$VB_SOURCES/${engine}.image.json" "$engine" "$tag" "$march" "$btype" \
           "$(image_digest "${rt_image}:${tag}")" "$(image_digest "${bn_image}:${tag}" 2>/dev/null || echo unbuilt)" <<'PY'
import json, sys
out, engine, tag, march, btype, rt, bn = sys.argv[1:8]
json.dump({"engine": engine, "tag": tag, "march": march, "build_type": btype,
           "runtime_image_id": rt, "bench_image_id": bn},
          open(out, "w"), indent=2, sort_keys=True)
open(out, "a").write("\n")
PY
}

case "$ENGINE" in
  # `all` stays the three baseline engines. Extra versions such as mariadb123
  # are opt-in, because each is another hour of compiling that nobody asked for
  # by typing "all".
  all)        build_engine mariadb; build_engine alisql; build_engine pgvector ;;
  mariadb123) build_engine mariadb123 ;;
  mariadb|alisql|pgvector) build_engine "$ENGINE" ;;
  *) die "unknown engine: $ENGINE" ;;
esac

ok "images built"
