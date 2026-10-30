#!/usr/bin/env bash
#
# Entrypoint for the vector-bench MySQL 9.7 + MyVector runtime image.
#
#   vb-entrypoint server [extra mysqld args...]   start the server (init first run)
#   vb-entrypoint init                            initialise the data directory only
#   vb-entrypoint client [args...]                open a client on the unix socket
#   vb-entrypoint <anything else>                 exec it verbatim
#
# A copy of entrypoint-percona.sh in shape -- both are MySQL 9.7.2 trees, so
# initialisation is `mysqld --initialize-insecure` -- with three additions that
# exist because MyVector is a PLUGIN and not part of the server:
#
# 1. THE PLUGIN IS LOADED WITH --plugin-load-add ON EVERY START, not with
#    INSTALL PLUGIN once. Its system variables (`myvector_index_dir`,
#    `myvector_config_file`, `myvector_index_bg_threads`) are then ordinary
#    command-line options, so the values in force are the ones in the `exec`
#    line below and in the manifest's server flags -- the property every other
#    engine's knobs have.
#
# 2. THE PLUGIN'S SQL OBJECTS ARE CREATED ONCE, AT INITIALISATION, from the
#    vendor's own sql/myvectorplugin.sql minus its INSTALL PLUGIN line (the
#    Dockerfile strips it). They are the UDFs (`myvector_ann_set`,
#    `myvector_row_distance`, ...), the `mysql.myvector_columns` view and the
#    `MYVECTOR_INDEX_*` procedures, and they persist in the data directory. The
#    script uses DELIMITER, which is a client directive, so it cannot go in
#    --init-file; a short-lived socket-only server runs it through the client.
#
# 3. THE PLUGIN CONNECTS BACK TO THIS SERVER, and needs credentials to do it.
#    `MYVECTOR_INDEX_BUILD` opens its own connection to read the table
#    (`BuildMyVectorIndexSQL`, also behind `myvector_index_refresh`), and the
#    online-update listener opens a replication stream
#    (`myvector_binlog_loop`; started by the plugin, not relied on by this
#    framework -- see docs/04-engine-notes.md). Both read `myvector.cnf`
#    (`readConfigFile`), which the plugin refuses unless its mode is 0600 or
#    0400 -- and a refused or missing file makes the build return an error
#    ROW, not an SQL error. It is written here, over the unix socket, as the
#    bench account (which init.sql grants ALL, so it has the REPLICATION
#    CLIENT/SLAVE and SELECT the listener needs). The server runs as root in
#    this container, which the plugin's ownership check exempts.

set -euo pipefail

ROOT_DIR="${VB_ROOT_DIR:-/opt/myvector}"
DATA_DIR="${VB_DATA_DIR:-/var/lib/vbench/data}"
SOCKET="${VB_SOCKET:-/var/run/vbench/myvector.sock}"
LOG_FILE="${VB_LOG_FILE:-/var/lib/vbench/myvector.err}"
INIT_SQL="${VB_INIT_SQL:-/opt/vbench/init.sql}"
PLUGIN_SQL="${VB_PLUGIN_SQL:-$ROOT_DIR/share/myvectorplugin.sql}"
# Outside the MySQL data directory on purpose but inside the same volume: the
# harness sees it read-only beside the data directory, and DROP DATABASE never
# meets files it does not own.
INDEX_DIR="${VB_MYVECTOR_INDEX_DIR:-/var/lib/vbench/myvector-index}"
CNF_FILE="${VB_MYVECTOR_CNF:-/var/lib/vbench/myvector.cnf}"
DB_USER="${VB_DB_USER:-bench}"
DB_PASSWORD="${VB_DB_PASSWORD:-bench}"

log() { printf '[vb-myvector] %s\n' "$*" >&2; }

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
CLIENT="$(find_bin mysql)" || { log "FATAL: no mysql client under $ROOT_DIR"; exit 1; }

user_args() { [[ "$(id -u)" -eq 0 ]] && printf '%s' "--user=root"; }

ensure_dirs() {
  mkdir -p "$DATA_DIR" "$(dirname "$SOCKET")" "$(dirname "$LOG_FILE")" "$INDEX_DIR"
}

write_cnf() {
  # Rewritten on every start so a changed password cannot leave a stale file.
  # umask first, chmod after: the file must never exist with wider permissions,
  # even for the instant between creation and chmod.
  ( umask 077
    printf 'myvector_user_id=%s\nmyvector_user_password=%s\nmyvector_socket=%s\n' \
      "$DB_USER" "$DB_PASSWORD" "$SOCKET" > "$CNF_FILE" )
  chmod 0600 "$CNF_FILE"
}

plugin_args() {
  printf '%s\n' \
    "--plugin-load-add=myvector.so" \
    "--myvector-index-dir=$INDEX_DIR" \
    "--myvector-config-file=$CNF_FILE"
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

  # Register the plugin's SQL objects (note 2). Socket only, no binlog, and
  # `myvector_feature_level=3` -- bit 0 set disables the plugin's binlog
  # thread (`myvector_binlog_loop`: `if (myvector_feature_level & 1)`), which
  # would otherwise spend this bootstrap retrying a connection.
  local boot_sock="/var/run/vbench/bootstrap.sock"
  log "registering MyVector UDFs, view and procedures from $PLUGIN_SQL"
  local -a boot=( --no-defaults --datadir="$DATA_DIR" --socket="$boot_sock"
                  --skip-networking --skip-log-bin --log-error="$LOG_FILE"
                  --pid-file=/var/run/vbench/bootstrap.pid
                  --myvector-feature-level=3 )
  mapfile -t _pa < <(plugin_args)
  boot+=( "${_pa[@]}" )
  local u; u="$(user_args)"; [[ -n "$u" ]] && boot+=( "$u" )
  "$MYSQLD" "${boot[@]}" &
  local pid=$! i
  for i in $(seq 1 120); do
    "$CLIENT" -uroot --socket="$boot_sock" -e 'SELECT 1' >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { log "FATAL: bootstrap server exited"; tail -50 "$LOG_FILE" >&2; exit 1; }
    sleep 0.5
  done
  "$CLIENT" -uroot --socket="$boot_sock" < "$PLUGIN_SQL"
  "$CLIENT" -uroot --socket="$boot_sock" < "$INIT_SQL"
  # Proof the objects exist, not a hope: a missing UDF would otherwise surface
  # as "FUNCTION myvector_ann_set does not exist" on the first query.
  "$CLIENT" -uroot --socket="$boot_sock" -N -e \
    "SELECT COUNT(*) FROM mysql.func WHERE name IN ('myvector_ann_set','myvector_row_distance','myvector_search_open_udf')" \
    | grep -qx 3 || { log "FATAL: MyVector UDFs were not registered"; exit 1; }
  "$CLIENT" -uroot --socket="$boot_sock" -e 'SHUTDOWN'
  wait "$pid" || true
  log "initialisation complete"
}

start_server() {
  ensure_dirs
  initialise
  write_cnf

  local -a args=(
    --no-defaults
    --datadir="$DATA_DIR"
    --socket="$SOCKET"
    --log-error="$LOG_FILE"
    --pid-file=/var/run/vbench/myvector.pid
    --skip-name-resolve
  )
  mapfile -t _pa < <(plugin_args)
  args+=( "${_pa[@]}" )
  [[ -f "$INIT_SQL" ]] && args+=( --init-file="$INIT_SQL" )
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
  client) exec "$CLIENT" --socket="$SOCKET" "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
