#!/usr/bin/env bash
#
# Materialise engine source trees at pinned tags into vector-bench/sources/
# and build the per-engine Docker build contexts under work/buildctx/.
#
# The vendor repositories in the parent directory are treated as READ-ONLY.
# We never check out, fetch into, or otherwise modify them:
#
#   * AliSQL has no submodules, so `git archive <tag>` straight out of the
#     local repo is enough and needs no network at all.
#   * MariaDB needs the `libmariadb` submodule, which is NOT populated in the
#     local checkout. `git archive` cannot include submodules, so for MariaDB we
#     clone into sources/ using the local repo as an object `--reference`
#     (fast, almost no network for the main history) and then initialise only
#     the submodules the build actually requires.
#
# Usage:
#   scripts/prepare-sources.sh [--engine mariadb|alisql|pgvector|all] [--force]

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ENGINE=all
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

need_cmd git tar python3
mkdir -p "$VB_SOURCES" "$VB_BUILDCTX"
assert_not_vendor_repo "$VB_SOURCES"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# record_meta <engine> <tag> <sha> <origin>
record_meta() {
  local engine="$1" tag="$2" sha="$3" origin="$4"
  python3 - "$VB_SOURCES/${engine}.source.json" "$engine" "$tag" "$sha" "$origin" <<'PY'
import json, sys
out, engine, tag, sha, origin = sys.argv[1:6]
with open(out, "w") as fh:
    json.dump({"engine": engine, "tag": tag, "commit": sha, "origin": origin},
              fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
  ok "$engine: tag=$tag commit=${sha:0:12} origin=$origin"
}

# Reuse an existing tarball when the pinned commit has not changed.
tarball_current() {
  local engine="$1" want_sha="$2" tar="$3"
  [[ $FORCE -eq 0 ]] || return 1
  [[ -f "$tar" ]] || return 1
  local meta="$VB_SOURCES/${engine}.source.json"
  [[ -f "$meta" ]] || return 1
  local have
  have="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("commit",""))' "$meta")"
  [[ "$have" == "$want_sha" ]]
}

# ---------------------------------------------------------------------------
# AliSQL — no submodules, pure offline `git archive`
# ---------------------------------------------------------------------------

prepare_alisql() {
  local tag; tag="$(yq_get "$VB_CONFIG/engines/alisql.yml" source.tag AliSQL-8.0.44-2)"
  local upstream; upstream="$(yq_get "$VB_CONFIG/engines/alisql.yml" source.upstream https://github.com/alibaba/AliSQL.git)"
  local tar="$VB_SOURCES/alisql-${tag}.tar"
  local repo="$VB_REPO_ALISQL" origin=local

  if [[ ! -d "$repo/.git" ]] || ! git -C "$repo" rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
    warn "AliSQL tag $tag not available locally; cloning from $upstream"
    repo="$VB_SOURCES/alisql-git"
    origin=upstream
    if [[ ! -d "$repo/.git" ]]; then
      git clone --quiet --depth 1 --branch "$tag" "$upstream" "$repo"
    fi
  fi

  local sha; sha="$(git -C "$repo" rev-parse "refs/tags/$tag^{commit}")"
  if tarball_current alisql "$sha" "$tar"; then
    ok "alisql: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
    stage_context alisql "$tar"; return
  fi

  info "alisql: exporting $tag from $repo (read-only git archive)"
  # storage/duckdb/third_parties is excluded: the DuckDB engine is switched off
  # in the build, and shipping it would add hundreds of MB to the build context.
  git -C "$repo" archive --format=tar --prefix=source/ "$tag" \
      -- . ':(exclude)storage/duckdb/third_parties' > "$tar.tmp"
  mv "$tar.tmp" "$tar"
  record_meta alisql "$tag" "$sha" "$origin"
  stage_context alisql "$tar"
}

# ---------------------------------------------------------------------------
# MariaDB — needs the libmariadb submodule, so clone (referencing local objects)
# ---------------------------------------------------------------------------

prepare_mariadb() {
  # Takes the engine name so additional MariaDB versions (mariadb123, ...)
  # reuse this routine. Tarball and checkout names carry the tag, so two
  # versions never collide.
  local engine="${1:-mariadb}"
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local tag; tag="$(yq_get "$cfg" source.tag mariadb-11.8.8)"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/MariaDB/server.git)"
  local submods; submods="$(yq_get "$cfg" source.submodules libmariadb)"
  local tar="$VB_SOURCES/mariadb-${tag}.tar"
  local checkout="$VB_SOURCES/mariadb-${tag}"

  local want_sha=""
  if [[ -d "$VB_REPO_MARIADB/.git" ]] \
     && git -C "$VB_REPO_MARIADB" rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
    want_sha="$(git -C "$VB_REPO_MARIADB" rev-parse "refs/tags/$tag^{commit}")"
  fi

  if [[ -n "$want_sha" ]] && tarball_current "$engine" "$want_sha" "$tar"; then
    ok "$engine: source tarball up to date ($(human_bytes "$(stat -c%s "$tar")"))"
    stage_context "$engine" "$tar"; return
  fi

  if [[ ! -d "$checkout/.git" ]]; then
    rm -rf "$checkout"
    local ref_args=()
    if [[ -d "$VB_REPO_MARIADB/.git" ]]; then
      # Reuse the local object store so the clone costs almost no network.
      # --dissociate makes the result independent of the reference afterwards,
      # so the vendor repo can be moved or deleted without breaking us.
      info "$engine: cloning $tag referencing local objects at $VB_REPO_MARIADB"
      ref_args=(--reference-if-able "$VB_REPO_MARIADB" --dissociate)
    else
      info "$engine: cloning $tag from $upstream (no local repo to reference)"
    fi
    git clone --quiet --branch "$tag" --single-branch "${ref_args[@]}" \
        "$upstream" "$checkout"
  else
    info "$engine: reusing existing checkout at $checkout"
  fi

  local sha; sha="$(git -C "$checkout" rev-parse HEAD)"

  # Only the submodules the build actually needs. ColumnStore, RocksDB, wsrep,
  # wolfssl, libmarias3 and duckdb are all disabled in the Dockerfile, so
  # pulling them would waste gigabytes.
  local IFS=,
  for sm in $submods; do
    [[ -n "$sm" ]] || continue
    if [[ -z "$(ls -A "$checkout/$sm" 2>/dev/null)" ]]; then
      info "$engine: initialising submodule $sm"
      git -C "$checkout" submodule update --init --depth 1 -- "$sm" \
        || die "failed to initialise submodule $sm (network required once)"
    fi
  done
  unset IFS

  info "$engine: packing build context tarball"
  tar --create --file "$tar.tmp" \
      --exclude-vcs --exclude='.git' \
      --transform='s,^\.,source,' \
      -C "$checkout" .
  mv "$tar.tmp" "$tar"
  record_meta "$engine" "$tag" "$sha" "$checkout"
  stage_context "$engine" "$tar"
}

# ---------------------------------------------------------------------------
# pgvector — small extension, cloned at its release tag
# ---------------------------------------------------------------------------

prepare_pgvector() {
  local cfg="$VB_CONFIG/engines/pgvector.yml"
  local tag; tag="$(yq_get "$cfg" source.tag v0.8.6)"
  local upstream; upstream="$(yq_get "$cfg" source.upstream https://github.com/pgvector/pgvector.git)"
  local checkout="$VB_SOURCES/pgvector-${tag}"
  local tar="$VB_SOURCES/pgvector-${tag}.tar"

  if [[ ! -d "$checkout/.git" ]]; then
    rm -rf "$checkout"
    info "pgvector: cloning $tag from $upstream"
    git clone --quiet --depth 1 --branch "$tag" "$upstream" "$checkout"
  fi
  local sha; sha="$(git -C "$checkout" rev-parse HEAD)"

  if tarball_current pgvector "$sha" "$tar"; then
    ok "pgvector: source tarball up to date"
    stage_context pgvector "$tar"; return
  fi

  git -C "$checkout" archive --format=tar --prefix=source/ HEAD > "$tar.tmp"
  mv "$tar.tmp" "$tar"
  record_meta pgvector "$tag" "$sha" "$upstream"
  stage_context pgvector "$tar"
}

# ---------------------------------------------------------------------------
# Build context staging
# ---------------------------------------------------------------------------

# Docker build contexts are per-engine directories holding exactly one source
# tarball plus that engine's auxiliary files. Keeping them separate means a
# MariaDB rebuild never has to ship AliSQL's 1.5 GB of source to the daemon.
stage_context() {
  local engine="$1" tar="$2"
  local ctx="$VB_BUILDCTX/$engine"
  mkdir -p "$ctx"
  # Hardlink when possible (same filesystem, zero copy), else fall back to copy.
  rm -f "$ctx/source.tar"
  ln "$tar" "$ctx/source.tar" 2>/dev/null || cp "$tar" "$ctx/source.tar"
  # Files shared by every engine image (the pinned Python stack), then the
  # engine's own auxiliary files (entrypoint, confs) which may override them.
  find "$VB_DOCKER/_shared" -maxdepth 1 -type f \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  # Auxiliary files (entrypoint, init.sql, confs) come from the engine's own
  # docker/ directory, or from the engine it aliases. Without the alias lookup
  # a second MariaDB version compiles for an hour and then dies on
  #   COPY failed: stat entrypoint-mariadb.sh: file does not exist
  # because docker/mariadb123/ does not exist and never needs to.
  local base; base="$(yq_get "$VB_CONFIG/engines/$engine.yml" alias_of "$engine")"
  find "$VB_DOCKER/$base" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  if [[ ! -f "$ctx/source.tar" ]]; then
    die "$engine: build context is missing source.tar at $ctx"
  fi
  ok "$engine: build context ready at $ctx (aux files from docker/$base)"
}

# ---------------------------------------------------------------------------

prepare_mongodb() {
  # The one engine with nothing to compile.
  #
  # Percona Search ships as tarballs, packages and images only, so there is no
  # pinned tag and commit to record and no source.tar to stage. What replaces
  # them is the image digest: a tag floats, and a run built from a floating tag
  # is unreproducible in exactly the dimension this script exists to guarantee.
  # Both digests are resolved here and written into the build context, where
  # the build reads them and the manifest records them.
  local engine=mongodb
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local ctx="$VB_BUILDCTX/$engine"
  local mongot server
  mongot="$(yq_get "$cfg" source.mongot_image "")"
  server="$(yq_get "$cfg" source.server_image "")"
  [[ -n "$mongot" && -n "$server" ]] || die "$engine: source.mongot_image / source.server_image not set in $cfg"

  mkdir -p "$ctx"
  say "$engine: resolving image digests (nothing is compiled for this engine)"

  local digests="$ctx/image-digests.txt"
  : > "$digests"
  local ref name digest
  for ref in "$server" "$mongot"; do
    docker pull --quiet "$ref" >/dev/null || die "$engine: cannot pull $ref"
    digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "$ref" 2>/dev/null || true)"
    [[ -n "$digest" ]] || die "$engine: $ref has no repo digest; push it to a registry or pin it by hand"
    name="$([[ "$ref" == "$server" ]] && echo server || echo mongot)"
    printf '%s=%s\n' "$name" "$digest" >> "$digests"
    ok "$engine: $name pinned to $digest"
  done

  find "$VB_DOCKER/_shared" -maxdepth 1 -type f -exec cp {} "$ctx/" \; 2>/dev/null || true
  find "$VB_DOCKER/$engine" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  ok "$engine: build context ready at $ctx (no source.tar: nothing is built from source)"
}

# ---------------------------------------------------------------------------

prepare_valkey() {
  # Nothing is fetched or compiled: Percona ships both the server and the
  # search module as packages, and the build installs them from the repository
  # named in the engine config. What this stages is the build context and the
  # package list the Dockerfile checks before installing, so a name that does
  # not resolve fails with the repository's actual contents rather than after
  # the corpus is loaded.
  local engine=valkey
  local cfg="$VB_CONFIG/engines/$engine.yml"
  local ctx="$VB_BUILDCTX/$engine"
  local repo; repo="$(yq_get "$cfg" source.repository "")"
  [[ -n "$repo" ]] || die "$engine: source.repository not set in $cfg"

  mkdir -p "$ctx"
  say "$engine: provenance is the $repo repository (nothing is compiled)"
  printf '%s\n' "$repo" > "$ctx/percona-repository.txt"

  find "$VB_DOCKER/_shared" -maxdepth 1 -type f -exec cp {} "$ctx/" \; 2>/dev/null || true
  find "$VB_DOCKER/$engine" -maxdepth 1 -type f ! -name 'Dockerfile' \
       -exec cp {} "$ctx/" \; 2>/dev/null || true
  ok "$engine: build context ready at $ctx (no source.tar: installed from packages)"
}


case "$ENGINE" in
  all)        prepare_mariadb mariadb; prepare_alisql; prepare_pgvector ;;
  mariadb)    prepare_mariadb mariadb ;;
  mariadb123) prepare_mariadb mariadb123 ;;
  alisql)     prepare_alisql ;;
  pgvector)   prepare_pgvector ;;
  mongodb)    prepare_mongodb ;;
  valkey)     prepare_valkey ;;
  *) die "unknown engine: $ENGINE (expected mariadb|mariadb123|alisql|pgvector|mongodb|valkey|all)" ;;
esac

ok "sources prepared"
