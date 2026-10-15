#
# Percona Server (MySQL 9.7.1 base + native InnoDB HNSW vector index),
# built from a pinned SHA on the `vector-mvp` branch.
#
# Targets:
#   runtime  — server binaries only. Use this to run Percona Server by hand
#              (see docs/03-running-manually.md).
#   bench    — runtime + the Python stack ann-benchmarks drives.
#
# Build context is work/buildctx/percona/, produced by scripts/prepare-sources.sh.
#
# REVIEW STATUS: this file has never been built. docs/09-adding-percona-server.md
# §7 lists what the first build has to prove. Every flag below cites the line on
# the pinned tree that made it necessary; none was guessed from a commit subject.
#
# Differences from docker/alisql/Dockerfile, which is otherwise the closest
# template (both are MySQL-derived, both initialise with
# `mysqld --initialize-insecure`, both create the bench account from --init-file):
#
#   1. OUT-OF-SOURCE BUILD. AliSQL's own build.sh forces an in-source build.
#      Percona Server refuses one: CMakeLists.txt:583 is
#      `OPTION(FORCE_INSOURCE_BUILD "Allow in-source build" OFF)` and :606 says
#      "You *can* force in-source build by invoking cmake with
#      -DFORCE_INSOURCE_BUILD=1". We do not force it; cmake runs in /build.
#
#   2. NO SUBMODULE FETCH, WHICH TAKES FOUR FLAGS AND NOT TWO. The pinned
#      tree's .gitmodules lists four submodules, and prepare-sources.sh ships a
#      `git archive`, which omits submodule paths entirely. Every one of them
#      therefore has to be switched off, or cmake descends into a directory
#      that is not in the tarball:
#
#        storage/rocksdb/rocksdb  -> -DWITH_ROCKSDB=OFF
#        extra/coredumper         -> -DWITH_COREDUMPER=OFF
#                                    CMakeLists.txt:1558, default ON
#        extra/jwt-cpp            -> -DWITH_AUTH_OPENID_CONNECT=OFF
#                                    CMakeLists.txt:167, default ON
#        extra/libkmip            -> -DWITHOUT_COMPONENT_KEYRING_KMIP=1
#
#      The last one is the reason this list is four lines long. libkmip is not
#      behind an OPTION() with a default; it is behind a two-symbol condition
#      that fires when NEITHER symbol is defined — CMakeLists.txt:2295-2297:
#
#        IF(NOT DEFINED WITH_COMPONENT_KEYRING_KMIP AND NOT DEFINED WITHOUT_COMPONENT_KEYRING_KMIP
#           OR WITH_COMPONENT_KEYRING_KMIP)
#          ADD_SUBDIRECTORY(extra/libkmip)
#        ENDIF()
#
#      AND binds tighter than OR, so with neither defined that is
#      (TRUE AND TRUE) OR FALSE = TRUE — which is the original defect: an
#      earlier revision of this file enumerated all four submodules correctly,
#      said "two", and passed three flags, so the build would have failed at
#      configure time on a directory the tarball does not contain.
#
#      -DWITH_COMPONENT_KEYRING_KMIP=OFF closes the gate too. Measured, not
#      reasoned: the condition above was transcribed verbatim into a standalone
#      project and run under cmake 3.28.3 —
#
#        (no flags)                          -> WOULD_ADD_SUBDIRECTORY
#        -DWITH_COMPONENT_KEYRING_KMIP=OFF   -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=0     -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=NO    -> SKIPPED
#        -DWITH_COMPONENT_KEYRING_KMIP=ON    -> WOULD_ADD_SUBDIRECTORY
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=1  -> SKIPPED
#        -DWITHOUT_COMPONENT_KEYRING_KMIP=0  -> SKIPPED
#
#      An earlier revision of this note had that backwards, on the argument
#      that defining the WITH_ symbol leaves it defined. That conflated
#      `DEFINED WITH_X` with `WITH_X`: defining it is exactly what makes the
#      AND false, and the OR's second operand is `WITH_X` evaluated as a
#      boolean, where OFF, 0 and NO are CMake false constants. The component's
#      own guard agrees — components/keyrings/keyring_kmip/CMakeLists.txt:28 is
#      `IF(NOT WITH_COMPONENT_KEYRING_KMIP) RETURN()`, keyed off the WITH_
#      spelling. Reasoning about a tool's semantics when the tool is installed
#      and the answer is one command away is the mirror image of this round's
#      other lesson about quoted hedges.
#
#      WITHOUT_ is still the better choice here, for the reason the last row of
#      that table shows: the gate tests `DEFINED WITHOUT_...` and never its
#      value, so WITHOUT_ closes it on definedness alone and cannot be broken
#      by a value spelling. WITH_=OFF relies on false-constant evaluation,
#      which is correct but is one more thing to be right about.
#
#      The alternative (listing them in percona.yml's `source.submodules:`, the
#      way mariadb.yml lists libmariadb) would make the build fetch from three
#      more hosts for plugins no benchmark here exercises. With the four flags
#      below this build needs no network beyond the base image and apt — the
#      same property docker/alisql/Dockerfile relies on.
#
#   3. BOOST NEEDS NO FLAG AND NO DOWNLOAD. cmake/boost.cmake:26-27 reads
#      "# Always use the bundled version." / SET(BOOST_SOURCE_DIR
#      ${CMAKE_SOURCE_DIR}/extra/boost), and the pinned tree carries
#      extra/boost/boost_1_87_0. So there is no -DWITH_BOOST and no
#      -DDOWNLOAD_BOOST here, deliberately.
#
#   4. NO mysql_native_password ANYWHERE. See init.sql — the plugin is gone from
#      the server in this base and requesting it is a hard error.
#
#   5. WE DO NOT USE build-ps/build-binary.sh, AND THE REASON IS THE BENCHMARK
#      AND NOT CONVENIENCE. It is the obvious thing to reach for -- it is the
#      vendor's own build, so it must work -- and it was audited against the
#      pinned tree on 2026-10-14 rather than dismissed. Six of its choices
#      disqualify it here, and all six are read from the script at the pin:
#
#        :17, :55   TARGET_CFLAGS='' and the only -march it ever sets is
#                   `-m32 -march=i686`, for 32-bit. On x86_64 the flags come
#                   from `rpm --eval %optflags` (:217-218) inside
#                   `if [ -n "$(command -v rpm)" ]`, so on this noble image
#                   there is no rpm, COMMON_FLAGS stays empty, and CFLAGS
#                   (:233) carries nothing. Percona would compile with no
#                   -march while the other four source-built engines carry
#                   x86-64-v3 -- the first invariant in CLAUDE.md, and the one
#                   that turns the study into a comparison of compiler flags.
#        :270       WITH_PGO defaults to 1: a three-pass profile-guided build.
#                   Percona would be the only engine in the study with PGO.
#        :312       -DWITH_LTO=ON, against a tree default of OFF
#                   (CMakeLists.txt:1017, OPTION at :1020). Same objection.
#        :311       -DWITH_PERCONA_TELEMETRY=ON, against a tree default of OFF
#                   (CMakeLists.txt:2392). Periodic background work inside a
#                   server whose CPU and wall time this benchmark attributes to
#                   query execution. ON is Percona's PACKAGING choice.
#        :299, :313 -DWITH_ROCKSDB=ON and -DWITH_JS_LANG=ON with
#                   -DV8_LIB_DIR=${WITH_V8}/... — the rocksdb submodule note 2
#                   explains we do not ship, and a prebuilt V8 tree.
#        :308       -DFORCE_INSOURCE_BUILD=1, which note 1 exists to avoid.
#
#      Two of its flags are also simply dead on this tree, which is the other
#      half of why the list cannot be copied: `-DWITH_LIBEVENT=bundled` (:309)
#      and `-DWITH_INNODB_MEMCACHED=ON` (:302). Measured, not inferred --
#      `extra/libevent` and `plugin/innodb_memcached` are both 404 at this
#      commit, and neither symbol appears in the top-level CMakeLists.txt or
#      in any file under cmake/. cmake warns about unused -D variables instead
#      of failing, so copying them would be silent noise. Anyone re-deriving
#      this flag list from the vendor script will reach for them;
#      tests/test_e6_percona_vendor_alignment.py fails if they land.
#
#      What IS taken from the vendor is everything that is about making it
#      compile rather than about how fast it runs: the builder apt list (note
#      on it below), and `bundled` for the three third-party libraries where
#      this tree offers the choice -- curl (:304), zlib (:303) and zstd
#      (:310). zstd was already the default (cmake/zstd.cmake:25); zlib was
#      `system` here until this commit and is `bundled` in cmake/zlib.cmake's
#      own default too (:117). Bundled is the path Percona's CI exercises, and
#      every one of the five configure failures on 2026-10-14 was a missing
#      system library.
#
#      Vendor packages deliberately NOT taken, with the reason, so the audit
#      does not have to be redone: valgrind, gdb, doxygen, doxygen-gui,
#      graphviz, devscripts, debhelper, fakeroot, ccache, patchelf,
#      python3-mysqldb, libeatmydata, dh-apparmor and copyright-update are
#      their packaging and test pipeline, not compile inputs; libsystemd-dev
#      and libjemalloc-dev govern options this tree defaults OFF
#      (CMakeLists.txt:1655 and :874); libmecab-dev likewise, since
#      build-binary.sh passes -DWITH_MECAB only when asked (:84); and
#      `libpam-dev` is the virtual name whose provider we install directly --
#      see the test that exists for exactly that mistake.
#

ARG BASE_IMAGE=ubuntu:24.04

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder
# ---------------------------------------------------------------------------
ARG BUILD_TYPE=RelWithDebInfo
ARG MARCH=x86-64-v3
ARG CMAKE_FLAGS=""
ARG EXTRA_CXXFLAGS=""
ARG JOBS=0

# NOT the same shape as the AliSQL builder stage any more, and the difference
# is measured. `file` is used by the strip pass at the end; `python3` is
# required by the MySQL build itself (comp_err and the generated-source steps),
# not by the benchmark. Everything from libldap2-dev onwards is this tree's
# own requirement, found on 2026-10-14 by building it for the first time.
#
# AliSQL builds from the same MySQL lineage with none of them, so this is not
# a MySQL-family cost -- it is what Percona Server 9.0 plus its own components
# need. Five configure failures, in the order they happened:
#
#   CMakeLists.txt:2075   -DWITH_AUTHENTICATION_LDAP=ON, but missing system
#                         libraries  (fed by the LDAP, SASL and KERBEROS
#                         warnings together -- any one of the three trips it)
#   CMakeLists.txt:2127   -DWITH_CURL=system, but missing system libraries,
#                         preceded by an ALIAS failure for `ext::curl` at
#                         cmake/curl.cmake:246. WITH_CURL defaults to "system"
#                         on every UNIX (CMakeLists.txt:998-1002).
#   cmake/rpc.cmake:108   Could not find rpc/rpc.h in /usr/include or
#                         /usr/include/tirpc            -> libtirpc-dev
#   plugin/auth_ldap/CMakeLists.txt:21
#                         Required LDAP dev library not found. A PERCONA
#                         plugin, and the half of that which matters is that
#                         no upstream -DWITH_AUTHENTICATION_* flag governs it.
#                         The check is NOT unconditional, as an earlier
#                         revision of this note claimed: it sits inside
#                         `IF(WITH_PERCONA_AUTHENTICATION_LDAP)` at
#                         plugin/auth_ldap/CMakeLists.txt:16, declared at
#                         CMakeLists.txt:2004-2011 with the default ON
#                         everywhere but APPLE. So the governing flag exists;
#                         it is simply a different, Percona-specific one from
#                         the family we were setting, and
#                         -DWITH_PERCONA_AUTHENTICATION_LDAP=OFF would have
#                         disabled it.
#   components/keyrings/keyring_vault/CMakeLists.txt:33
#                         could not find CURL library, after -DWITH_CURL=0
#                         had been set to clear an earlier failure
#
#   CMakeLists.txt:2075   again, with libsasl2-dev installed and LDAP found:
#                         `Cannot find SASL development libraries` is what
#                         cmake/sasl.cmake prints, and the dev package is not
#                         what it was missing. MYSQL_CHECK_SASL looks for the
#                         SCRAM *plugin* -- FIND_LIBRARY(NAMES scram
#                         PATH_SUFFIXES sasl2) -- and returns without warning
#                         only when it finds one. libsasl2-dev ships nothing
#                         under sasl2/ at all; the unversioned
#                         sasl2/libscram.so symlink FIND_LIBRARY needs comes
#                         from libsasl2-modules, a RUNTIME package. The
#                         warning's own install hint names
#                         libsasl2-modules-gssapi-mit beside the dev package
#                         for this reason, and the vendor's install_deps lists
#                         both. Measured by unpacking all three .debs for
#                         noble, after a --no-cache rebuild proved the dev
#                         package was installed and the check failed anyway.
#
# EVERY ONE OF THESE HAS AN OFF SWITCH SOMEWHERE. An earlier revision of this
# note said only rpc.cmake did not, and that was wrong three times over. The
# switches are recorded so the strategy below reads as a choice rather than as
# the absence of one:
#
#   :2075              -DWITH_AUTHENTICATION_LDAP=OFF, and it was tried
#   :2127              -DWITH_CURL=0, and it was tried -- see keyring_vault
#   auth_ldap:21       -DWITH_PERCONA_AUTHENTICATION_LDAP=OFF, per above
#   keyring_vault:33   components/keyrings/keyring_vault/CMakeLists.txt:25 is
#                      `option(WITH_COMPONENT_KEYRING_VAULT ... ON)` with an
#                      early `return()` at :26-29, so =OFF closes it
#   rpc.cmake:108      -DWITH_TIRPC=bundled takes the branch at
#                      cmake/rpc.cmake:51-59, which sets TIRPC_FOUND itself
#                      and calls ADD_SUBDIRECTORY(extra/tirpc) -- and
#                      extra/tirpc IS in the tarball, since .gitmodules at
#                      this pin lists only the four submodules note 2 names
#
# UNPROVEN, AND DELIBERATELY LEFT THAT WAY: the vendor's own
# build-ps/percona-server-9.0_builder.sh:603-609 carries a comment asserting
# there is "no bundled-tirpc fallback despite extra/tirpc/ existing in the
# tree". That contradicts the branch cited on the last row above. Nothing here
# can settle it -- no build of this image has ever succeeded, so neither
# reading has been executed -- and it must not be resolved by argument.
#
# The strategy does not depend on which of those is right. Every flag that was
# tried moved the failure instead of removing it, and -DWITH_CURL=0 is the
# clearest case because it CAUSED keyring_vault:33; each further switch costs
# a multi-hour build to evaluate. Hence the rule recorded beside cmake_flags
# in config/engines/percona.yml: on this tree, switching a dependency off
# breaks a Percona component that wanted the target, while installing the
# dependency does not.
#
# WHY NO CONSUMER-SIDE FLAG COULD EVER HAVE SAVED THE CURL ONE, which is the
# sharpest thing here and the easiest to get backwards: the curl check at
# CMakeLists.txt:2121-2131 sits at TOP LEVEL, inside no IF() at all. It runs
# whether or not anything in the build wants curl, and :2127 is fatal on the
# library being absent rather than on anything needing it. OpenID Connect
# genuinely is a curl consumer -- plugin/auth_openid_connect/CMakeLists.txt:32-34
# links `curl`, inside `IF(WITH_AUTH_OPENID_CONNECT)` at :16 -- so the
# -DWITH_AUTH_OPENID_CONNECT=OFF in the cmake list below really does remove
# one, and it still does not help.
#
# The list is this tree's own build-ps/percona-server-9.0_builder.sh
# `install_deps()`, filtered to the development packages and plugins among its
# entries and to what noble actually has -- the vendor list also names gcc-8
# through gcc-15, valgrind, doxygen, python-mysqldb, dh-systemd, libzstd1-dev
# and libncurses5-dev, several of which have no candidate here. It is NOT "the
# subset that gates configure", which an earlier revision of this note claimed
# while also calling it a superset: it is a SUPERSET and it was not pruned.
# Only three entries are tied to a failure recorded above -- libldap2-dev,
# libtirpc-dev and libsasl2-modules. With -DWITH_CURL=bundled and the vendor's
# bundled libevent and zstd (build-binary.sh:309-310), libcurl4-openssl-dev,
# liblz4-dev, libzstd-dev, libevent-dev and libre2-dev are probably
# unnecessary; nothing here can run a build to find out which, so none were
# removed. Prune it against a build that succeeded, not against reasoning.
#
# libpam0g-dev AND NOT THE VENDOR'S `libpam-dev`, for a reason an earlier
# revision of this note also got wrong: `libpam-dev` does not resolve to
# nothing on noble -- it is a VIRTUAL package whose only provider is
# libpam0g-dev, and packages.ubuntu.com files it under "virtual". Naming the
# real package is still the right call, because a purely virtual name is one
# more thing for apt to be clever about, but no failure above asked for it:
# CMakeLists.txt:165 is `OPTION(WITH_PAM "Build with Percona PAM plugin" OFF)`,
# DEFAULT OFF, and only build-binary.sh:298 turns it on. libpam0g-dev is
# precautionary, like most of this list.
#
# TELEMETRY, recorded for whoever next compares this against build-binary.sh:
# that script passes -DWITH_PERCONA_TELEMETRY=ON at :311 and this build passes
# nothing at all. CMakeLists.txt:2392 is
# `option(WITH_PERCONA_TELEMETRY "Build Percona Telemetry component" OFF)` --
# already OFF -- so no telemetry component is built and no flag is needed to
# achieve that. Leave it alone rather than "fixing" the difference: a
# component doing periodic background work inside a server whose CPU and wall
# time this benchmark attributes to query execution is a confound, and ON is a
# choice Percona's PACKAGING makes, not the source default.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake bison pkg-config \
      libssl-dev libncurses-dev libudev-dev libaio-dev \
      libnuma-dev zlib1g-dev libedit-dev \
      python3 file \
      libldap2-dev libsasl2-dev libkrb5-dev libcurl4-openssl-dev libtirpc-dev \
      libpam0g-dev liblz4-dev libzstd-dev libevent-dev libre2-dev libwrap0-dev \
      libsasl2-modules libsasl2-modules-gssapi-mit \
      perl gawk unzip curl

WORKDIR /src
COPY source.tar /tmp/source.tar
RUN tar -xf /tmp/source.tar -C /src && rm -f /tmp/source.tar

# Flags mirror config/engines/percona.yml. -march is the only one that may ever
# differ between engines, and it must not: all seven engines compile SIMD
# distance kernels, and a different flag on one turns the benchmark into a
# comparison of compiler flags. The value is written to /opt/percona/.march so
# the ann overlay module can report which one the image actually used — see
# overlay/.../percona/module.py, which overrides _image_march() for exactly this.
#
# `make --output-sync` is a diagnosability flag and it is not optional.
# The first `build --engines percona` ever run ended with nothing more useful
# than
#
#   make: *** [Makefile:166: all] Error 2
#
# as its final line. The compiler diagnostic and the `[ 26%] Building CXX
# object ...` line that names the file it came from are written by two
# different processes, so above -j1 they land apart, and the sub-make's own
# `make[1]: *** Error 1` lands apart from the top-level `Error 2` as well.
# Measured on GNU Make 4.3, a two-level recursive build at -j4 with one target
# failing: without the flag the `error:` text sits three unrelated progress
# lines below the target that produced it, with three more after it before the
# top-level error; with `=target` it is the very next line. This image is built
# at `nproc` jobs by default, which on the reference host is 80.
#
# `=target` and not `=recurse`: `recurse` groups output per recursive make, and
# CMake's generated top-level Makefile recurses once into CMakeFiles/Makefile2
# for the whole build, so `=recurse` would buffer every line until the build
# ended -- no progress at all for the hours this takes. `=line` keeps progress
# live but still splits a multi-line diagnostic from its own target.
#
# `|| make -j1` is the other half, and it is about where the error lands
# rather than whether it is adjacent to its file. `--output-sync` puts
# the compiler diagnostic next to the `[ 26%] Building CXX object ...` line
# that names the file; it does NOT move either of them to the end of the log.
# When a target fails at -j80 make stops launching new work but lets the ~79
# already-running targets finish, and their blocks print AFTER the failing
# one -- so the last twenty lines of the log are the stragglers draining out,
# and three builds of this image were reported from that tail with the error
# never read. The serial retry resumes from wherever the parallel build got
# to, recompiles only what is missing, stops at the first error and prints it
# as the last thing in the log. It costs minutes rather than hours because
# everything before the failure is already built.
#
# Two consequences to know rather than discover:
#
#   * a failure that was a parallel-build RACE will not recur serially, so
#     the retry succeeds and the image builds. That is a better outcome than
#     failing, and it produces the same binary -- but it means a green build
#     here does not prove the parallel build is sound.
#   * `set -e` does not fire between the two, because `||` suppresses errexit
#     for its left side. If the serial retry also fails the chain still exits
#     non-zero, which is what fails the Docker build.
#
# The retry is spelled `--jobs=1` and not `-j1`, which is not a style choice.
# `make -j` is the region anchor two other suites split this file on, and it
# has to stay UNIQUE: with a second `make -j` token in the same command, the
# guard in tests/test_e5_percona_make_output_sync.py stops failing loudly
# when the flag is misplaced and starts silently falling through to the
# retry, widening the region past the invocation it is meant to bound.
# Measured: with `-j1` the "wrong side of the anchor" probe went from
# reporting an empty region to reporting a populated one, which is the
# failure the probe exists to detect. `--jobs=1` is identical to make.
#
# The flag goes AFTER -j"$JOBS" deliberately. `make -j` is the literal string
# tests/test_e1_percona_design.py and tests/test_e4_percona_build_flags.py
# split this file on to bound the cmake flag region, and they do not survive
# its loss equally: test_e4 asserts the anchor is present and fails loudly,
# while test_e1:350-351 splits without that assertion, so a missing separator
# silently widens its region from the cmake invocation to end of file -- every
# assertion still green over a bound that no longer exists. Measured, by
# running both parses over a copy of this file with the flag moved ahead of
# -j: the guarded region goes from every flag to none at all, while the
# unguarded one keeps every flag and swallows the rest of the file with them,
# as far down as `ARG PERCONA_TAG`. That asymmetry is why
# tests/test_e5_percona_make_output_sync.py pins the flag's side of the
# anchor rather than trusting either suite to notice.
# CFLAGS carries an explicit -D_FORTIFY_SOURCE=2. That is a build FIX, not a
# hardening preference: without it this image does not compile at all.
#
# The symptom was three builds ending in `make: *** [Makefile:166: all] Error
# 2` with nothing in the tail naming a file. The serial retry above is what
# finally put the real error last:
#
#     extra/libfido2/libfido2-1.15.0/src/.../aes256.c.o
#     <command-line>: error: "_FORTIFY_SOURCE" redefined
#     <command-line>: note: this is the location of the previous definition
#
# Two definitions with different values reach the preprocessor. The tree asks
# for 2 -- source CMakeLists.txt:1586 prepends `-DNDEBUG -D_FORTIFY_SOURCE=2`
# for RELEASE/RELWITHDEBINFO/MINSIZEREL -- and gcc on noble supplies 3 of its
# own. Measured on ubuntu:24.04's gcc 13.3: `echo | gcc -O3 -dM -E -` prints
# `#define _FORTIFY_SOURCE 3`.
#
# Ubuntu's default stands down as soon as it sees a -D for the macro on the
# command line: measured, `gcc -O3 -D_FORTIFY_SOURCE=2` warns nothing and the
# effective value is 2. So the tree's own definition is arriving in a form the
# driver's scan cannot see. `-Wp,-D...` is the only form reproduced that
# behaves this way -- plain -D, a @response-file, C++ and -std=c99 all let the
# default stand down. The emitting site was NOT found in the pinned tree and
# this comment does not pretend to know it: the fix does not depend on which
# site it is. A plain -D here makes the driver stand down, and every
# definition that then remains says 2. Identical redefinition is silent.
#
# Only libfido2 FAILS on it. The rest of the build prints the same
# redefinition as a warning and carries on -- bundled absl does, at 69% --
# which is why three builds reached 20-69% before dying. libfido2 adds
# `-pedantic-errors` (source extra/libfido2/libfido2-1.15.0/CMakeLists.txt
# :329-330), and that turns this one diagnostic into a hard error.
# -pedantic-errors is also the only flag tried that reproduces the log's exact
# form: a bare `error:` with a `note:` and no `cc1: all warnings being treated
# as errors` line. It is NOT -Werror, which prints a `[-Werror]` tag and that
# summary. Nor can it be demoted -- -Wno-error=cpp, -Wno-pedantic and
# -Wno-builtin-macro-redefined all leave the error standing (measured).
# Removing the conflict is the only route, which is why this is a -D and not
# a -Wno-.
#
# The value is 2 and must not be 3: it has to AGREE with what the tree asks
# for, and 3 against the tree's 2 would reinstate the conflict it removes. It
# also lines this image up with the two MariaDB builds, which ask for 2
# plainly and get it (mariadb-11.8.8:CMakeLists.txt:290 and
# mariadb-12.3.2:CMakeLists.txt:303, read at both pins) while this one was
# compiling at 3. _FORTIFY_SOURCE=3 adds dynamic object-size checks that 2
# does not, so that was a codegen difference between engines in the same
# study -- the same class of problem as -march. The pinned AliSQL sets no
# level of its own and is still on the compiler default; that is a separate
# question and this file does not answer it.
#
# tests/test_e7_percona_fortify_level.py holds the value and the reasoning.
RUN set -eux; \
    JOBS="${JOBS:-0}"; [ "$JOBS" -gt 0 ] 2>/dev/null || JOBS="$(nproc)"; \
    mkdir -p /build; \
    cd /build; \
    export CFLAGS="-O3 -g -fexceptions -fno-strict-aliasing -fno-omit-frame-pointer -march=${MARCH} -D_FORTIFY_SOURCE=2 ${EXTRA_CXXFLAGS}"; \
    export CXXFLAGS="$CFLAGS"; \
    cmake /src/source \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX=/opt/percona \
      -DMYSQL_DATADIR=/var/lib/vbench/data \
      -DSYSCONFDIR=/opt/percona \
      -DMYSQL_SERVER_SUFFIX="-vector-bench" \
      -DWITH_COREDUMPER=OFF \
      -DWITH_AUTH_OPENID_CONNECT=OFF \
      -DWITH_CURL=bundled \
      -DWITH_ROCKSDB=OFF \
      -DWITHOUT_COMPONENT_KEYRING_KMIP=1 \
      -DWITH_NDB=OFF \
      -DWITH_ROUTER=OFF \
      -DWITH_MYSQLX=OFF \
      -DWITH_UNIT_TESTS=OFF \
      -DWITH_SSL=system \
      -DWITH_ZLIB=bundled \
      -DWITH_ZSTD=bundled \
      ${CMAKE_FLAGS} ; \
    make -j"$JOBS" --output-sync=target || make --jobs=1; \
    make install; \
    echo "${MARCH}" > /opt/percona/.march

RUN find /opt/percona -type f -executable -exec file {} \; \
      | awk -F: '/ELF .* not stripped/{print $1}' | xargs -r strip --strip-debug

# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server runtime"
ARG PERCONA_TAG=unknown
ENV VB_ENGINE=percona \
    VB_ENGINE_TAG=${PERCONA_TAG} \
    VB_ROOT_DIR=/opt/percona \
    VB_DATA_DIR=/var/lib/vbench/data \
    VB_SOCKET=/var/run/vbench/percona.sock \
    PATH=/opt/percona/bin:$PATH

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libssl3t64 libncurses6 libaio1t64 libnuma1 libudev1 zlib1g libedit2 \
      ca-certificates procps tzdata \
    && useradd --system --create-home --home-dir /var/lib/vbench --shell /usr/sbin/nologin mysql 2>/dev/null || true

COPY --from=builder /opt/percona /opt/percona
COPY entrypoint-percona.sh /usr/local/bin/vb-entrypoint
COPY init.sql /opt/vbench/init.sql
RUN chmod +x /usr/local/bin/vb-entrypoint \
    && mkdir -p "${VB_DATA_DIR}" /var/run/vbench \
    && chown -R mysql:mysql /var/lib/vbench /var/run/vbench \
    && chmod 0644 /opt/vbench/init.sql

ENV VB_INIT_SQL=/opt/vbench/init.sql \
    VB_DB_USER=bench \
    VB_DB_PASSWORD=bench

EXPOSE 3306
ENTRYPOINT ["/usr/local/bin/vb-entrypoint"]
CMD ["server"]

# ---------------------------------------------------------------------------
FROM runtime AS bench
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="vector-bench Percona Server + ann-benchmarks"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential git \
      libmariadb-dev libhdf5-dev pkg-config

# The same client stack as the MariaDB and AliSQL bench images, for the same
# reason: MariaDB Connector/C speaks the MySQL protocol, so all three
# MySQL-family engines are driven through an identical client with
# binary-protocol prepared statements, and differences in the numbers belong to
# the servers. requirements-mysql.txt pins mariadb==1.1.10 with --no-binary.
#
# MARIADB_CONFIG points at the distro's mariadb_config and not at the one on
# PATH, exactly as in the MariaDB and AliSQL images. Here PATH starts with
# /opt/percona/bin, which carries no mariadb_config at all, so the pin is
# belt-and-braces rather than load-bearing — but it keeps the three images
# compiling the extension against one Connector/C, which is the point.
#
# ONE UNPROVEN THING, and it is the auth handshake rather than the build:
# this base has no mysql_native_password (see init.sql), so the bench account
# authenticates with caching_sha2_password. Over the unix socket the ann pass
# uses (vb_mysql.py passes unix_socket=) that is a cleartext exchange the server
# permits. Over the TCP connection the ops harness uses
# (mysql_family.py passes host=/port=) it is not, and Connector/C must either
# negotiate TLS or fetch the server's RSA public key. docs/09 §7 item 2 lists this as
# the first thing the smoke run has to settle.
COPY requirements-bench.txt requirements-mysql.txt /tmp/
RUN MARIADB_CONFIG=/usr/bin/mariadb_config \
    pip3 install --no-cache-dir --break-system-packages \
        -r /tmp/requirements-bench.txt -r /tmp/requirements-mysql.txt \
    && rm -f /tmp/requirements-bench.txt /tmp/requirements-mysql.txt \
    && python3 -c "import mariadb; print('Connector/Python', mariadb.__version__, 'OK')"

WORKDIR /home/app
ENV PERCONA_ROOT_DIR=/opt/percona \
    PERCONA_DB_WORKSPACE=/var/lib/vbench \
    DO_INIT_PERCONA=yes
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
