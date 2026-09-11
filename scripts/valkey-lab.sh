#!/usr/bin/env bash
#
# Run one script against a throwaway Valkey, without a benchmark run.
#
# The probe and verify scripts both need what an ops unit sets up -- a Valkey
# server with the search module, a private network, and a bench container that
# can import the harness -- and until now getting that meant hand-starting two
# containers with the right image names, environment and limits, or running a
# forty-hour profile to reach a five-minute question.
#
# Everything here is torn down on exit, including on Ctrl-C. Nothing touches
# results/ and nothing is recorded: this is for answering a question, not for
# producing a measurement.
#
#   scripts/valkey-lab.sh probe-valkey-filter.py --rows 200000
#   scripts/valkey-lab.sh verify-valkey-churn.py --rows 100000
#   scripts/valkey-lab.sh --shell            # a prompt, for poking by hand
#
# --memory sets both the container limit and maxmemory, which the server needs
# in bytes. Default is 32g, enough for 200,000 x 1536 with the graph and room
# to spare; raise it for --rows 990000.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MEMORY="32g"
SCRIPT=""
SHELL_MODE=0
declare -a SCRIPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory) MEMORY="$2"; shift 2 ;;
    --memory=*) MEMORY="${1#*=}"; shift ;;
    --shell) SHELL_MODE=1; shift ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) if [[ -z "$SCRIPT" && $SHELL_MODE -eq 0 ]]; then SCRIPT="$1"; else SCRIPT_ARGS+=("$1"); fi; shift ;;
  esac
done

[[ -n "$SCRIPT" || $SHELL_MODE -eq 1 ]] || die "name a script under scripts/, or pass --shell"

need_docker
RUNTIME_IMAGE="vector-bench/valkey-runtime"
BENCH_IMAGE="vector-bench/valkey-bench"
for image in "$RUNTIME_IMAGE" "$BENCH_IMAGE"; do
  image_exists "$image" || die "$image not found. Build it: ./run-benchmark.sh build --engines valkey"
done

if [[ $SHELL_MODE -eq 0 ]]; then
  [[ -f "${VB_ROOT}/scripts/${SCRIPT}" ]] || die "no such script: scripts/${SCRIPT}"
fi

STAMP="$(date -u '+%H%M%S')"
NET="vb-lab-${STAMP}"
SRV="vb-lab-valkey-${STAMP}"
CLI="vb-lab-client-${STAMP}"

cleanup() {
  docker rm -f "$CLI" "$SRV" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# maxmemory below the container limit: hitting the container limit gets the
# process OOM-killed, hitting maxmemory under noeviction returns an error that
# says what happened. Same 0.9 the resource profile uses.
MEM_BYTES="$(numfmt --from=iec "${MEMORY^^}")"
MAXMEMORY="$(( MEM_BYTES * 9 / 10 ))"

info "network ${NET}, ${MEMORY} container / $(human_bytes "$MAXMEMORY") maxmemory"
docker network create --label vector-bench=1 "$NET" >/dev/null

# noeviction is not optional. Under any other policy a full Valkey drops keys,
# and vectors vanishing mid-probe looks exactly like a bad index.
docker run -d --name "$SRV" --label vector-bench=1 \
  --network "$NET" --memory "$MEMORY" \
  -e VB_SERVER_ARGS="--maxmemory ${MAXMEMORY} --maxmemory-policy noeviction --io-threads 8" \
  -e VB_MAXMEMORY_BYTES="$MAXMEMORY" \
  "$RUNTIME_IMAGE" server >/dev/null

info "waiting for the search module"
for _ in $(seq 1 120); do
  if docker exec "$SRV" sh -c 'valkey-cli MODULE LIST 2>/dev/null | grep -qi search'; then
    ok "valkey up with valkey-search"
    break
  fi
  sleep 1
done || true
docker exec "$SRV" sh -c 'valkey-cli MODULE LIST 2>/dev/null | grep -qi search' \
  || die "valkey-search never loaded; docker logs ${SRV}"

if [[ $SHELL_MODE -eq 1 ]]; then
  info "server is ${SRV}, reachable as host '${SRV}' from this shell"
  docker run -it --rm --name "$CLI" --label vector-bench=1 --network "$NET" \
    -v "${VB_ROOT}/harness:/opt/harness:ro" \
    -v "${VB_ROOT}/scripts:/opt/scripts:ro" \
    -e PYTHONPATH=/opt -w /opt "$BENCH_IMAGE" bash
  exit 0
fi

info "running ${SCRIPT}"
# harness/ and scripts/ are mounted rather than baked, so an edit takes effect
# on the next invocation without rebuilding an image.
docker run --rm --name "$CLI" --label vector-bench=1 --network "$NET" \
  -v "${VB_ROOT}/harness:/opt/harness:ro" \
  -v "${VB_ROOT}/scripts:/opt/scripts:ro" \
  -e PYTHONPATH=/opt -e PYTHONUNBUFFERED=1 -w /opt \
  "$BENCH_IMAGE" python3 "/opt/scripts/${SCRIPT}" \
  --host "$SRV" "${SCRIPT_ARGS[@]}"
