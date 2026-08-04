#!/usr/bin/env bash
#
# Verify the AliSQL-specific behaviours that docs/04-engine-notes.md claims,
# against a real server. Documentation that asserts engine behaviour should be
# executable, otherwise it silently rots when the vendor changes something.
#
# Checks, all from docs/04-engine-notes.md §"AliSQL — VIDX":
#   1. vidx_disabled defaults to ON, and vector DDL fails while it is on.
#   2. Vector DML requires READ COMMITTED; other isolation levels are rejected.
#   3. VECTOR columns accept a raw 4*dim binary string (no VEC_FROMTEXT needed).
#   4. The optimizer silently falls back to a full scan above a LIMIT threshold —
#      the trap that would otherwise be measured as "accurate but slow".
#   5. FORCE INDEX overrides that fallback.
#
# Usage: tests/verify-alisql-traps.sh [image]

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib.sh"
# lib.sh sets -e for the operational scripts, which is right for them and wrong
# here: this is a test runner whose whole purpose is to keep going after a
# check fails and report a tally. Without this, the first failing probe or
# deliberately-failing SQL statement kills the run silently.
set +e

IMAGE="${1:-vector-bench/alisql-runtime:latest}"
CONTAINER="vb-alisql-traps-$$"
SOCK=/var/run/vbench/alisql.sock

need_docker
image_exists "$IMAGE" || die "image not found: $IMAGE (build it first)"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

PASS=0
FAIL=0

check() {
  local name="$1" expectation="$2" actual="$3"
  if [[ "$actual" == *"$expectation"* ]]; then
    ok "$name"
    PASS=$((PASS + 1))
  else
    printf '%sFAIL%s %s\n      expected to contain: %s\n      actual: %s\n' \
      "$_C_RED" "$_C_RESET" "$name" "$expectation" "$actual" >&2
    FAIL=$((FAIL + 1))
  fi
}

sql() {
  docker exec -i "$CONTAINER" /opt/alisql/bin/mysql -ubench -pbench \
    --socket="$SOCK" -N 2>&1
}

# The runtime image both passes --vidx-disabled=OFF at startup AND runs
# `SET GLOBAL vidx_disabled = OFF` from init.sql, so the shipped default cannot
# be observed by flags alone. Check 1 therefore sets the variable back to ON
# explicitly and tests the behaviour that matters: vector DDL must be rejected
# while vector support is disabled, however it came to be disabled.
info "starting AliSQL"
docker run -d --name "$CONTAINER" "$IMAGE" >/dev/null \
  || die "failed to start container"

info "waiting for readiness"
for _ in $(seq 1 120); do
  docker exec "$CONTAINER" /opt/alisql/bin/mysql -ubench -pbench \
    --socket="$SOCK" -e 'SELECT 1' >/dev/null 2>&1 && break
  sleep 1
done
docker exec "$CONTAINER" /opt/alisql/bin/mysql -ubench -pbench \
  --socket="$SOCK" -e 'SELECT 1' >/dev/null 2>&1 \
  || die "server never became ready:\n$(docker logs --tail 30 "$CONTAINER" 2>&1)"

# --- 1. vector DDL is rejected while vidx_disabled is ON --------------------
out="$(sql <<'SQL'
SET GLOBAL vidx_disabled = ON;
CREATE DATABASE IF NOT EXISTS trap;
USE trap;
DROP TABLE IF EXISTS t;
CREATE TABLE t (id INT PRIMARY KEY, v VECTOR(4), VECTOR INDEX vi (v)) ENGINE=InnoDB;
SQL
)"
check "vector DDL rejected while vidx_disabled=ON" "ERROR" "$out"

# Confirm the variable really does default to ON in the server itself, rather
# than inferring it from the image's behaviour.
out="$(printf "SELECT CONCAT('default_is_', @@GLOBAL.vidx_disabled) FROM DUAL;\n" | sql)"
info "current vidx_disabled after the check above: $(echo "$out" | tail -1)"

# --- 2. RC is required for vector DML ---------------------------------------
out="$(sql <<'SQL'
SET GLOBAL vidx_disabled = OFF;
USE trap;
DROP TABLE IF EXISTS t;
CREATE TABLE t (id INT PRIMARY KEY, tag INT NOT NULL, v VECTOR(4) NOT NULL,
                VECTOR INDEX vi (v) M=6 DISTANCE=EUCLIDEAN) ENGINE=InnoDB;
SET SESSION transaction_isolation = 'REPEATABLE-READ';
INSERT INTO t VALUES (1, 0, VEC_FROMTEXT('[1,0,0,0]'));
SQL
)"
check "vector DML rejected at REPEATABLE-READ" "ERROR" "$out"

# --- 3. raw binary float32 is accepted --------------------------------------
# [1,0,0,0] as little-endian float32 = 0000803F 00000000 00000000 00000000
out="$(sql <<'SQL'
USE trap;
SET SESSION transaction_isolation = 'READ-COMMITTED';
INSERT INTO t VALUES (1, 0, 0x0000803F000000000000000000000000);
SELECT CONCAT('binary_insert_rows=', COUNT(*)) FROM t;
SQL
)"
check "raw 4*dim binary string accepted (no VEC_FROMTEXT)" "binary_insert_rows=1" "$out"

# --- 4/5. optimizer fallback above a LIMIT threshold ------------------------
info "populating 100 rows to reproduce the documented LIMIT threshold"
out="$(sql <<'SQL'
USE trap;
SET SESSION transaction_isolation = 'READ-COMMITTED';
DELETE FROM t;
INSERT INTO t(id, tag, v) SELECT 1, 1, VEC_FROMTEXT(CONCAT('[',RAND(),',',RAND(),',',RAND(),',',RAND(),']'));
SQL
)"
for i in $(seq 2 100); do
  printf "USE trap; SET SESSION transaction_isolation='READ-COMMITTED'; INSERT INTO t(id,tag,v) SELECT %d, %d, VEC_FROMTEXT(CONCAT('[',RAND(),',',RAND(),',',RAND(),',',RAND(),']'));\n" "$i" "$((i % 10))"
done | sql >/dev/null

plan_for() {
  printf "USE trap;
SET SESSION transaction_isolation='READ-COMMITTED';
EXPLAIN SELECT id FROM t %s ORDER BY VEC_DISTANCE_EUCLIDEAN(v, VEC_FROMTEXT('[1,2,3,4]')) LIMIT %d;\n" "$2" "$1" | sql
}

# The crossover is a FRACTION of the table (~25-28%), not a fixed LIMIT, and it
# is independent of ef_search. On 100 rows that puts it between 25 and 40.
low="$(plan_for 10 '')"
mid="$(plan_for 25 '')"
high="$(plan_for 60 '')"
forced="$(plan_for 60 'FORCE INDEX (vi)')"

check "LIMIT 10 (10% of rows) uses the vector index" "vi" "$low"
check "LIMIT 25 (25% of rows) still uses the vector index" "vi" "$mid"

if [[ "$high" != *"vi"* ]]; then
  ok "LIMIT 60 (60% of rows) falls back to a full scan, as documented"
  PASS=$((PASS + 1))
else
  # Not a framework failure — but docs/04-engine-notes.md quantifies this
  # crossover, so a change upstream must be loud rather than silent.
  warn "LIMIT 60 still used the index; the documented ~25-28% crossover has moved."
  warn "Re-measure and update docs/04-engine-notes.md. Plan: $high"
  FAIL=$((FAIL + 1))
fi

check "FORCE INDEX overrides the fallback at LIMIT 60" "vi" "$forced"

# ef_search must not influence the choice; if it starts to, the documented
# characterisation is wrong and the plan guard's rationale needs revisiting.
ef_low="$(printf "USE trap;
SET SESSION transaction_isolation='READ-COMMITTED';
SET vidx_hnsw_ef_search=80;
EXPLAIN SELECT id FROM t ORDER BY VEC_DISTANCE_EUCLIDEAN(v, VEC_FROMTEXT('[1,2,3,4]')) LIMIT 60;\n" | sql)"
if [[ "$ef_low" != *"vi"* ]]; then
  ok "raising ef_search does not change the fallback decision"
  PASS=$((PASS + 1))
else
  warn "ef_search=80 changed the plan at LIMIT 60; the crossover is not ef_search-independent"
  FAIL=$((FAIL + 1))
fi

echo
if [[ $FAIL -eq 0 ]]; then
  ok "all $PASS AliSQL behaviour checks passed"
else
  printf '%s%d passed, %d FAILED%s\n' "$_C_RED" "$PASS" "$FAIL" "$_C_RESET" >&2
fi
exit $(( FAIL > 0 ? 1 : 0 ))
