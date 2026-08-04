#!/usr/bin/env bash
#
# Entrypoint for the vector-bench AliSQL runtime image.
#
#   vb-entrypoint server [extra mysqld args...]   start the server (init first run)
#   vb-entrypoint init                            initialise the data directory only
#   vb-entrypoint client [args...]                open a client on the unix socket
#   vb-entrypoint <anything else>                 exec it verbatim
#
# AliSQL is MySQL 8.0.44 based, so initialisation is `mysqld --initialize-insecure`
# rather than MariaDB's mariadb-install-db.
#
# Note on vector support: VIDX ships DISABLED. The server is therefore started
# with --vidx-disabled=OFF by default here; without it every CREATE TABLE with a
# VECTOR column fails with ER_VECTOR_DISABLED. Vector DML additionally requires
# READ-COMMITTED, which the harness sets per session.

set -euo pipefail

ROOT_DIR="${VB_ROOT_DIR:-/opt/alisql}"
DATA_DIR="${VB_DATA_DIR:-/var/lib/vbench/data}"
SOCKET="${VB_SOCKET:-/var/run/vbench/alisql.sock}"
LOG_FILE="${VB_LOG_FILE:-/var/lib/vbench/alisql.err}"
INIT_SQL="${VB_INIT_SQL:-/opt/vbench/init.sql}"

log() { printf '[vb-alisql] %s\n' "$*" >&2; }

find_bin() {
  local name="$1" p
  for p in "$ROOT_DIR/bin/$name"; do
    [[ -x "$p" ]] && { printf '%s' "$p"; return 0; }
  done
  p="$(command -v "$name" 2>/dev/null || true)"
  [[ -n "$p" ]] && { printf '%s' "$p"; return 0; }
  return 1
}

MYSQLD="$(find_bin mysqld)" || { log "FATAL: no mysqld under $ROOT_DIR"; exit 1; }

user_args() { [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"; }

ensure_dirs() {
  mkdir -p "$DATA_DIR" "$(dirname "$SOCKET")" "$(dirname "$LOG_FILE")"
}

initialise() {
  if [[ -d "$DATA_DIR/mysql" ]]; then
    log "data directory already initialised at $DATA_DIR"
    return 0
  fi
  log "initialising data directory at $DATA_DIR"
  "$MYSQLD" \
    --no-defaults \
    --initialize-insecure \
    --datadir="$DATA_DIR" \
    --log-error="$LOG_FILE" \
    $(user_args) >&2
  log "initialisation complete"
}

start_server() {
  ensure_dirs
  initialise

  # --skip-grant-tables is deliberately absent: on MySQL 8 it implicitly enables
  # --skip-networking, which would make this server unreachable over TCP from
  # the harness container. The bench account is created from --init-file instead.
  local -a args=(
    --no-defaults
    --datadir="$DATA_DIR"
    --socket="$SOCKET"
    --log-error="$LOG_FILE"
    --pid-file=/var/run/vbench/alisql.pid
    --skip-name-resolve
    --vidx-disabled=OFF
  )
  # NOTE: no --mysqlx=0 here. The image is built with -DWITH_MYSQLX=OFF, so the
  # X plugin does not exist and the option is rejected outright:
  #   [ERROR] [MY-000067] [Server] unknown variable 'mysqlx=0'.
  # A server flag that is only valid for some build configurations does not
  # belong in the entrypoint; pass it via VB_SERVER_ARGS if you build with the
  # X plugin enabled.
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

  log "exec $MYSQLD ${args[*]}"
  exec "$MYSQLD" "${args[@]}"
}

cmd="${1:-server}"; shift || true
case "$cmd" in
  server) start_server "$@" ;;
  init)   ensure_dirs; initialise ;;
  client)
    CLIENT="$(find_bin mysql)" || { log "no client binary"; exit 1; }
    exec "$CLIENT" --socket="$SOCKET" "$@"
    ;;
  *) exec "$cmd" "$@" ;;
esac
