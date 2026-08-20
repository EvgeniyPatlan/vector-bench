#!/usr/bin/env bash
#
# Entrypoint for the vector-bench Valkey + valkey-search runtime image.
#
#   vb-entrypoint server [extra valkey args...]  start the server
#   vb-entrypoint client [args...]               open valkey-cli
#   vb-entrypoint <anything else>                exec it verbatim
#
# Refuses to start without the search module. A Valkey with no module accepts
# every write and then fails every FT.SEARCH, which in a long run shows up as
# an engine that loaded fine and scored zero.

set -euo pipefail

PORT="${VB_VALKEY_PORT:-6379}"
# Resolved by dpkg at build time and recorded in the image; the environment
# variable is only an override for running the image by hand.
MODULE="${VB_MODULE_PATH:-}"
if [[ -z "$MODULE" && -r /opt/valkey-artifacts/.module_path ]]; then
  MODULE="$(cat /opt/valkey-artifacts/.module_path)"
fi
MODULE="${MODULE:-/usr/lib/valkey/modules/libsearch.so}"
DATADIR="${VB_DATADIR:-/var/lib/vbench}"

log() { printf '[vb-valkey] %s\n' "$*" >&2; }

start_server() {
  if [[ ! -f "$MODULE" ]]; then
    log "ERROR: search module not found at $MODULE"
    log "modules present under /usr/lib/valkey:"
    dpkg -L percona-valkey-search 2>/dev/null | grep -E '\.so$' >&2 || \
      log "  (percona-valkey-search installed no shared object)"
    exit 1
  fi
  mkdir -p "$DATADIR"

  local -a args=(
    valkey-server
    --port "$PORT"
    --bind "0.0.0.0"
    --dir "$DATADIR"
    --protected-mode no
    --loadmodule "$MODULE"
  )
  # shellcheck disable=SC2206
  [[ -n "${VB_SERVER_ARGS:-}" ]] && args+=( ${VB_SERVER_ARGS} )
  args+=( "$@" )

  log "starting: ${args[*]}"
  exec "${args[@]}"
}

case "${1:-server}" in
  server) shift; start_server "$@" ;;
  client) shift; exec valkey-cli -p "$PORT" "$@" ;;
  *)      exec "$@" ;;
esac
