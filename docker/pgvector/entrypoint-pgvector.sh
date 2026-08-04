#!/usr/bin/env bash
#
# Entrypoint for the vector-bench PostgreSQL + pgvector runtime image.
#
#   vb-entrypoint server [extra postgres args...]  start the server (init first run)
#   vb-entrypoint init                             initialise the cluster only
#   vb-entrypoint client [args...]                 open psql on the ann database
#   vb-entrypoint <anything else>                  exec it verbatim
#
# Wraps the official postgres image entrypoint so that the CREATE EXTENSION step
# and the vector-bench server arguments are applied consistently, and so this
# image is driven the same way as the MySQL-family ones.

set -euo pipefail

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PGDB="${POSTGRES_DB:-ann}"
PGUSER_NAME="${POSTGRES_USER:-postgres}"

log() { printf '[vb-pgvector] %s\n' "$*" >&2; }

start_server() {
  local -a args=( postgres -c "listen_addresses=*" -c "unix_socket_directories=/var/run/postgresql" )
  # shellcheck disable=SC2206
  [[ -n "${VB_SERVER_ARGS:-}" ]] && args+=( ${VB_SERVER_ARGS} )
  args+=( "$@" )

  log "delegating to the official postgres entrypoint: ${args[*]}"
  # docker-entrypoint.sh handles initdb, the trust auth setup, creating
  # $POSTGRES_DB, and running /docker-entrypoint-initdb.d/* on first boot.
  exec docker-entrypoint.sh "${args[@]}"
}

case "${1:-server}" in
  server) shift || true; start_server "$@" ;;
  init)
    # Force the official entrypoint through its initdb path, then stop.
    log "initialising cluster at $PGDATA"
    docker-entrypoint.sh postgres --version >/dev/null
    ;;
  client) shift || true; exec psql -U "$PGUSER_NAME" -d "$PGDB" "$@" ;;
  *) exec "$@" ;;
esac
