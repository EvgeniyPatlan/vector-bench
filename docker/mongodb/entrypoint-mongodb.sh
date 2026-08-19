#!/usr/bin/env bash
#
# Entrypoint for the vector-bench Percona Server for MongoDB + mongot image.
#
#   vb-entrypoint server [extra mongod args...]  start mongod and mongot
#   vb-entrypoint init                           initialise the replica set only
#   vb-entrypoint client [args...]               open mongosh on the ann database
#   vb-entrypoint <anything else>                exec it verbatim
#
# Two processes, one container. Percona Search needs an initiated replica set
# (standalone is unsupported) and a mongot reachable on the port mongod is told
# about, so this brings both up in the right order and does not report ready
# until each is actually answering.

set -euo pipefail

DBPATH="${VB_DBPATH:-/var/lib/vbench/db}"
MONGOT_DATA="${VB_MONGOT_DATA:-/var/lib/vbench/mongot}"
RS="${VB_REPLICA_SET:-rs0}"
MONGOT_PORT="${VB_MONGOT_PORT:-27028}"
HEAP_GB="${VB_MONGOT_HEAP_GB:-8}"
PORT="${VB_MONGOD_PORT:-27017}"

log() { printf '[vb-mongodb] %s\n' "$*" >&2; }

start_mongot() {
  mkdir -p "$MONGOT_DATA"
  # Heap scales with the number of indexed fields, not the number of vectors:
  # the vectors are served from memory-mapped segments in the filesystem cache.
  # Oversizing it takes memory away from the cache that answers queries, which
  # MongoDB's own guidance warns against above 50% of available memory.
  log "starting mongot on ${MONGOT_PORT} with ${HEAP_GB}g heap"
  /opt/mongot/bin/mongot \
      --port "$MONGOT_PORT" \
      --dataDir "$MONGOT_DATA" \
      --mongodHostAndPort "localhost:${PORT}" \
      --jvm-flags "-Xms${HEAP_GB}g -Xmx${HEAP_GB}g" \
      >"${MONGOT_DATA}/mongot.log" 2>&1 &
  echo $! > /tmp/mongot.pid
}

wait_for_mongod() {
  for _ in $(seq 1 120); do
    if mongosh --quiet --port "$PORT" --eval 'db.adminCommand({ping:1}).ok' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  log "mongod did not answer on ${PORT}"; return 1
}

init_replica_set() {
  wait_for_mongod
  if mongosh --quiet --port "$PORT" --eval 'rs.status().ok' >/dev/null 2>&1; then
    log "replica set already initiated"; return 0
  fi
  log "initiating single-node replica set ${RS}"
  mongosh --quiet --port "$PORT" --eval \
    "rs.initiate({_id:'${RS}',members:[{_id:0,host:'localhost:${PORT}'}]})" >/dev/null
  # PRIMARY, not just initiated: writes fail until the election completes, and
  # a load that starts one second early fails on the first insert_many.
  for _ in $(seq 1 60); do
    if [[ "$(mongosh --quiet --port "$PORT" --eval 'db.hello().isWritablePrimary' 2>/dev/null)" == "true" ]]; then
      log "replica set is primary"; return 0
    fi
    sleep 1
  done
  log "replica set did not reach primary"; return 1
}

start_server() {
  mkdir -p "$DBPATH"
  local -a args=(
    mongod --dbpath "$DBPATH" --port "$PORT" --bind_ip_all
    --replSet "$RS"
  )
  # shellcheck disable=SC2206
  [[ -n "${VB_SERVER_ARGS:-}" ]] && args+=( ${VB_SERVER_ARGS} )
  args+=( "$@" )

  "${args[@]}" &
  local mongod_pid=$!

  init_replica_set
  start_mongot

  # mongod is the container's lifetime; if it exits, so does everything.
  wait "$mongod_pid"
}

case "${1:-server}" in
  server) shift; start_server "$@" ;;
  init)   wait_for_mongod && init_replica_set ;;
  client) shift; exec mongosh --port "$PORT" "${VB_DATABASE:-ann}" "$@" ;;
  *)      exec "$@" ;;
esac
