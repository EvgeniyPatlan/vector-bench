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
MODULE="${VB_MODULE_PATH:-/usr/lib/valkey/modules/libsearch.so}"
DATADIR="${VB_DATADIR:-/var/lib/vbench}"

log() { printf '[vb-valkey] %s\n' "$*" >&2; }

start_server() {
  if [[ ! -f "$MODULE" ]]; then
    log "ERROR: search module not found at $MODULE"
    log "modules present under /usr/lib/valkey:"
    find /usr/lib/valkey /usr/lib/valkey-search -name '*.so' 2>/dev/null >&2 || \
      log "  (none)"
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
