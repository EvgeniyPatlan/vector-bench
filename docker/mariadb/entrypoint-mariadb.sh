#!/usr/bin/env bash
#
# Entrypoint for the vector-bench MariaDB runtime image.
#
#   vb-entrypoint server [extra mariadbd args...]   start the server (init first run)
#   vb-entrypoint init                              initialise the data directory only
#   vb-entrypoint client [args...]                  open a client on the unix socket
#   vb-entrypoint <anything else>                   exec it verbatim
#
# Server arguments come from three places, applied in order:
#   1. the image defaults below
#   2. $VB_SERVER_ARGS  (whitespace-separated; set by the harness)
#   3. arguments passed on the command line
# Later wins, which is how MariaDB itself resolves duplicate options.

set -euo pipefail

ROOT_DIR="${VB_ROOT_DIR:-/opt/mariadb}"
DATA_DIR="${VB_DATA_DIR:-/var/lib/vbench/data}"
SOCKET="${VB_SOCKET:-/var/run/vbench/mariadb.sock}"
LOG_FILE="${VB_LOG_FILE:-/var/lib/vbench/mariadb.err}"
INIT_SQL="${VB_INIT_SQL:-/opt/vbench/init.sql}"

log() { printf '[vb-mariadb] %s\n' "$*" >&2; }

find_bin() {
  local name="$1" p
  for p in "$ROOT_DIR/bin/$name" "$ROOT_DIR/scripts/$name"; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  p="$(command -v "$name" 2>/dev/null || true)"
  [[ -n "$p" ]] && { printf '%s' "$p"; return 0; }
  return 1
}

MARIADBD="$(find_bin mariadbd || find_bin mysqld)" \
  || { log "FATAL: no mariadbd/mysqld under $ROOT_DIR"; exit 1; }
INSTALL_DB="$(find_bin mariadb-install-db || find_bin mysql_install_db || true)"

user_args() {
  # Running as uid 0 requires an explicit --user; running as anyone else must not
  # pass it, or the server refuses to start.
  [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"
}

ensure_dirs() {
  mkdir -p "$DATA_DIR" "$(dirname "$SOCKET")" "$(dirname "$LOG_FILE")"
}

initialise() {
  if [[ -d "$DATA_DIR/mysql" ]]; then
    log "data directory already initialised at $DATA_DIR"
    return 0
  fi
  [[ -n "$INSTALL_DB" ]] || { log "FATAL: mariadb-install-db not found"; exit 1; }
  log "initialising data directory at $DATA_DIR"
  "$INSTALL_DB" \
    --no-defaults \
    --skip-name-resolve \
    --skip-test-db \
    --auth-root-authentication-method=normal \
    --datadir="$DATA_DIR" \
    $(user_args) >&2
  log "initialisation complete"
}

start_server() {
  ensure_dirs
  initialise

  # NOTE: --skip-grant-tables is deliberately NOT used. On MySQL 8 (and so on
  # AliSQL) it implicitly enables --skip-networking, which would leave the
  # server unreachable over TCP. Both MySQL-family engines therefore start with
  # grants enabled and create the bench account from an --init-file, so their
  # startup and auth paths stay identical.
  local -a args=(
    --no-defaults
    --datadir="$DATA_DIR"
    --socket="$SOCKET"
    --log-error="$LOG_FILE"
    --pid-file=/var/run/vbench/mariadb.pid
    --skip-name-resolve
  )
  [[ -f "$INIT_SQL" ]] && args+=( --init-file="$INIT_SQL" )
  # --no-defaults must be the server's first argument; a duplicate arriving
  # later makes it exit 2. It is already in `args` above, so strip any copy
  # that comes in through VB_SERVER_ARGS rather than letting it break startup.
  if [[ -n "${VB_SERVER_ARGS:-}" ]]; then
    for _arg in ${VB_SERVER_ARGS}; do
      [[ "$_arg" == "--no-defaults" ]] && continue
      args+=( "$_arg" )
    done
  fi
  args+=( "$@" )
  local u; u="$(user_args)"; [[ -n "$u" ]] && args+=( "$u" )

  log "exec $MARIADBD ${args[*]}"
  exec "$MARIADBD" "${args[@]}"
}

cmd="${1:-server}"; shift || true
case "$cmd" in
  server) start_server "$@" ;;
  init)   ensure_dirs; initialise ;;
  client)
    CLIENT="$(find_bin mariadb || find_bin mysql)" || { log "no client binary"; exit 1; }
    exec "$CLIENT" --socket="$SOCKET" "$@"
    ;;
  *) exec "$cmd" "$@" ;;
esac
