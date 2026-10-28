#!/usr/bin/env python3
"""Turn a saved core file into a backtrace, with gdb inside the bench image.

Twelve of the twenty-four Percona ops units in run full-20261015-215006 died
inside the first INSERT -- exactly half. (`--profile full` is four datasets
times three M values, so twelve units a pass and twenty-four across both;
the twelve that died is the plan's count over the archived unit files, which
are not in this checkout. `config/engines/percona.yml`'s `error_log` note
still says "13 of 24"; that figure is wrong and the crash-capture change
corrects it.) The error log said the server went away and nothing
said why, because three separate things were missing: the server was not
asked for a core, teardown deleted the data directory any core would have
been written into, and gdb was in no image in this repository. The first two
are the crash-capture plumbing; this script is the third, and it is the step
that turns the artefact into something a human or a server team can read.

Diagnose, don't measure -- which is `lab`'s whole job. Nothing here is
recorded, no record is written, and no report can read this output.

    ./run-benchmark.sh lab --engine percona \\
        --artifacts results/<run-id> gdb-backtrace.py

`--artifacts` mounts that directory read-only at /artifacts, which is where
this script looks for the core. It has to be named explicitly: a lab
session's own run directory is `state/lab/<session>`, created empty moments
earlier, and the core is in `results/<run-id>/`, which the session cannot
name for itself.

WHAT THE OUTPUT WILL CONTAIN, because the honest answer is less than the
plan assumed. The image is built RelWithDebInfo, but the builder's last step
runs `strip --strip-debug` over every ELF under /opt/percona. That flag
removes every `.debug_*` section and keeps `.symtab`, so gdb resolves
function names and resolves nothing else. MEASURED on the build host
(binutils 2.42, gdb 15.1, Ubuntu 24.04 -- the image's own base) against a
toy program compiled `-g -O0` and then stripped the same way:

    with DWARF    #0  ... in inner (depth=5, seed=15) at c.c:2
                          local_a = 35 / local_b = 50 / p = 0x0
    after strip   #0  ... in inner ()
                          No symbol table info available.

So `thread apply all bt full` gives the call chain by function name across
every thread -- which is what attributes a crash to a code path, and is what
request 1 of docs/12-percona-feedback-plan.md asks for -- and no arguments,
no locals and no line numbers. If a variable's value is what the question
needs, that is a rebuild with the strip removed and several GB per image,
and it is a decision to make deliberately rather than a flag to add here.

THE CORE MAY SIMPLY NOT EXIST, and that is the likeliest outcome on a first
attempt rather than a defect in this script. A core is produced only if the
host's /proc/sys/kernel/core_pattern does not begin with `|` -- a pipe to
apport or systemd-coredump consumes it instead -- and that is the operator's
one-line opt-in, performed by hand, because a benchmark framework must not
modify the machine it measures.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

#: Where `cmd_lab --artifacts` mounts the directory holding the core.
ARTIFACTS = "/artifacts"

#: The server binary in this image. `CMAKE_INSTALL_PREFIX=/opt/percona` in
#: docker/percona/Dockerfile, and the bench stage is `FROM runtime AS bench`,
#: so the tree the runtime image carries is already here.
DEFAULT_BINARY = "/opt/percona/bin/mysqld"

#: The command. `thread apply all` and not a bare `bt`: the thread gdb stops
#: in is not reliably the one that crashed, and for a server with a thread
#: per connection plus InnoDB's background threads the interesting stack is
#: usually not the first one printed.
GDB_COMMAND = "thread apply all bt full"

#: Filenames a core may have. `core_pattern` is read, never written (see the
#: docstring), so the name is whatever the host decided -- `core`, `core.PID`
#: and a prefixed form are all ordinary -- and this has to search rather than
#: know. Order does not matter: anything matching any of these is a
#: candidate, and the caller is told when there is more than one.
CORE_GLOBS = ("core", "core.*", "*.core", "core-*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every lab script is handed --host and --port by OpsRun.run_script, so
    # accepting them is the contract. This one talks to no server: it reads a
    # file the server left behind, and the server it belonged to is gone.
    p.add_argument("--host", default=None,
                   help="accepted and ignored; this reads a file, not a server")
    p.add_argument("--port", default=None,
                   help="accepted and ignored, as --host")
    p.add_argument("--core", default=None,
                   help=f"the core file (default: search {ARTIFACTS})")
    p.add_argument("--binary", default=DEFAULT_BINARY,
                   help=f"the executable the core came from "
                        f"(default: {DEFAULT_BINARY})")
    p.add_argument("--list-cores", action="store_true",
                   help=f"print the candidates under {ARTIFACTS} and stop")
    return p.parse_args(argv)


def find_cores(root: str) -> list:
    """Every plausible core under `root`, largest first.

    Recursive because the crash-capture step files artefacts per phase, and
    largest first because a truncated core (RLIMIT_CORE, a full disk) sits
    beside a complete one often enough to be worth seeing -- and because the
    complete one is the one to open.
    """
    found = []
    for directory, _subdirs, names in os.walk(root):
        for name in names:
            if any(fnmatch.fnmatch(name, pattern) for pattern in CORE_GLOBS):
                path = os.path.join(directory, name)
                if os.path.isfile(path) and not os.path.islink(path):
                    found.append(path)
    return sorted(found, key=lambda path: -os.path.getsize(path))


def describe(path: str) -> str:
    """Path and size. The size is the point: a core truncated by RLIMIT_CORE
    or by a full disk opens and gives a short, wrong-looking stack."""
    size = os.path.getsize(path)
    if size >= 1024 ** 3:
        return f"{path}  ({size / (1024 ** 3):.2f} GiB)"
    return f"{path}  ({size / (1024 ** 2):.1f} MiB)"


def main(argv=None) -> int:
    args = parse_args(argv)

    gdb = shutil.which("gdb")
    if not gdb:
        print("no gdb on PATH in this image.\n"
              "It is installed in the BENCH stage of docker/percona/Dockerfile"
              ", so an\nimage built before that landed does not have it:\n\n"
              "  ./run-benchmark.sh build --engines percona\n",
              file=sys.stderr)
        return 1

    if not os.path.isdir(ARTIFACTS):
        print(f"{ARTIFACTS} is not mounted. Pass the directory holding the "
              f"core:\n\n  ./run-benchmark.sh lab --engine percona "
              f"--artifacts results/<run-id> \\\n"
              f"      {os.path.basename(__file__)}\n", file=sys.stderr)
        return 1

    candidates = find_cores(ARTIFACTS)
    if args.list_cores:
        for path in candidates:
            print(describe(path))
        if not candidates:
            print(f"no core under {ARTIFACTS}")
        return 0

    core = args.core
    if core is None:
        if not candidates:
            print(f"no core file under {ARTIFACTS}.\n"
                  f"Looked for {', '.join(CORE_GLOBS)} recursively. The two "
                  f"ordinary reasons:\n"
                  f"  * the host's /proc/sys/kernel/core_pattern begins with "
                  f"'|', so the kernel\n"
                  f"    piped the core to apport or systemd-coredump and no "
                  f"file was written;\n"
                  f"  * the phase did not crash, or ran before the server was "
                  f"asked for a core.\n", file=sys.stderr)
            return 1
        if len(candidates) > 1:
            print(f"{len(candidates)} cores under {ARTIFACTS}; name one with "
                  f"--core:", file=sys.stderr)
            for path in candidates:
                print(f"  {describe(path)}", file=sys.stderr)
            return 2
        core = candidates[0]

    if not os.path.isfile(core):
        print(f"no such core file: {core}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.binary):
        print(f"no such executable: {args.binary}\n"
              f"A core is only readable against the binary that wrote it; "
              f"pass --binary.", file=sys.stderr)
        return 1

    print(f"gdb      : {gdb}")
    print(f"binary   : {args.binary}")
    print(f"core     : {describe(core)}")
    print(f"command  : {GDB_COMMAND}")
    print("symbols  : function names only. The builder strips debug info "
          "with\n           `strip --strip-debug`, which keeps .symtab and "
          "removes DWARF, so\n           every frame will say \"No symbol "
          "table info available\" for its\n           locals. See this "
          "script's docstring for the measurement.\n", flush=True)

    # debuginfod off, for two reasons. It prompts on first use, and a prompt
    # in a --batch container run is a hang; and it fetches symbols over the
    # network from a host whose job is to measure, which is the same
    # objection CLAUDE.md makes to adding a host-side dependency. `--nx` for
    # a third: the working directory is /opt, with this repository's scripts
    # mounted under it, and a .gdbinit anywhere gdb looks would make the
    # output depend on something other than the core. On noble the system
    # /etc/gdb/gdbinit carries one comment line and no commands, so today it
    # changes nothing.
    command = [gdb, "--batch", "--nx",
               "-iex", "set debuginfod enabled off",
               "-ex", GDB_COMMAND,
               args.binary, core]
    # Output is inherited rather than captured: `bt full` over a large core
    # is slow, and an operator watching nothing for ten minutes cannot tell
    # it from a hang.
    completed = subprocess.run(command)
    if completed.returncode != 0:
        print(f"\ngdb exited {completed.returncode}", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
