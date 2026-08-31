"""Taking in a run measured on another machine.

Copying a directory into results/ already works; this is the same thing through
the browser, for when you have a .tar.gz and no shell on the box.

Extraction is the dangerous part. A tar archive can name absolute paths, walk
out with .., or carry a symlink pointing anywhere, and Python's own guard
(tarfile.data_filter) does not exist in this image's 3.11. So every member is
checked here rather than trusted, and the archive is unpacked into a temporary
directory and moved into place only once it is known to hold a run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MANIFEST = "run-manifest.json"

#: Refuse an archive larger than this before reading it into memory.
MAX_BYTES = 2 * 1024 ** 3
#: And refuse one that expands to more than this, which is the zip-bomb shape.
MAX_EXPANDED_BYTES = 20 * 1024 ** 3


class RejectedArchive(Exception):
    pass


def _check_member(member: tarfile.TarInfo, top: str) -> None:
    name = member.name
    if name.startswith("/") or os.path.isabs(name):
        raise RejectedArchive(f"absolute path in archive: {name}")
    parts = name.split("/")
    if ".." in parts:
        raise RejectedArchive(f"path escapes the archive root: {name}")
    if parts[0] != top:
        raise RejectedArchive(
            f"archive holds more than one top-level directory: {parts[0]} and {top}")
    if member.issym() or member.islnk():
        raise RejectedArchive(f"archive contains a link, which is not needed "
                              f"in a run directory: {name}")
    if member.isdev() or member.ischr() or member.isblk() or member.isfifo():
        raise RejectedArchive(f"archive contains a device or fifo: {name}")


def inspect(path: str) -> Tuple[str, List[tarfile.TarInfo]]:
    """The archive's single top-level directory and its validated members."""
    try:
        archive = tarfile.open(path, "r:*")
    except tarfile.TarError as exc:
        raise RejectedArchive(f"not a readable tar archive: {exc}") from None

    with archive:
        members = archive.getmembers()
        if not members:
            raise RejectedArchive("archive is empty")

        top = members[0].name.split("/")[0]
        if not NAME_RE.match(top):
            raise RejectedArchive(
                f"top-level directory {top!r} is not a usable run id "
                f"(letters, digits, dot, dash, underscore)")

        total = 0
        for member in members:
            _check_member(member, top)
            total += max(0, member.size)
        if total > MAX_EXPANDED_BYTES:
            raise RejectedArchive(
                f"archive expands to {total / 1024 ** 3:.1f} GB, which is more "
                f"than this accepts")

        names = {m.name for m in members}
        if f"{top}/{MANIFEST}" not in names:
            raise RejectedArchive(
                f"no {MANIFEST} in the archive. A run without its provenance "
                f"cannot be reported: the hardware, versions and limits behind "
                f"the numbers would be unknown.")
        return top, members


def import_bundle(results_dir: str, archive_path: str,
                  run_id: Optional[str] = None,
                  label: Optional[str] = None,
                  source: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """Unpack a run bundle into results/. Returns (run_id, errors)."""
    try:
        top, _members = inspect(archive_path)
    except RejectedArchive as exc:
        return None, [str(exc)]

    target_id = (run_id or top).strip()
    if not NAME_RE.match(target_id):
        return None, [f"{target_id!r} is not a usable run id: letters, digits, "
                      f"dot, dash and underscore, up to 64 characters"]

    destination = os.path.join(results_dir, target_id)
    if os.path.exists(destination):
        return None, [f"results/{target_id} already exists. Give it another "
                      f"name, or remove the existing one first."]

    # results/ is generated and gitignored, so on a checkout where nothing has
    # been run yet it does not exist. Creating the staging directory inside it
    # then raised FileNotFoundError out of this function, killed the request
    # handler, and closed the connection with no reply -- which the browser
    # reported as the upload dropping after every byte had arrived.
    try:
        os.makedirs(results_dir, exist_ok=True)
        staging = tempfile.mkdtemp(dir=results_dir, prefix=".import-")
    except OSError as exc:
        return None, [f"cannot write to {results_dir}: {exc}"]
    try:
        # 3.12 warns that 3.14 will filter by default, and 3.11 does not accept
        # the argument at all. Members are validated above either way; this only
        # settles which behaviour is asked for where it can be.
        extract_kwargs = {"filter": "tar"} if hasattr(tarfile, "data_filter") else {}

        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                _check_member(member, top)
                # Take the bytes, not the metadata. An archive's modes are its
                # author's choice, and honouring them lets one arrive
                # world-writable, setuid, or -- as a directory without +x --
                # unwritable enough that its own contents cannot be unpacked.
                member.mode = 0o755 if member.isdir() else 0o644
                member.uid, member.gid = os.getuid(), os.getgid()
                member.uname = member.gname = ""
                archive.extract(member, staging, **extract_kwargs)

        unpacked = os.path.join(staging, top)
        if not os.path.isfile(os.path.join(unpacked, MANIFEST)):
            return None, [f"no {MANIFEST} after unpacking"]

        _write_label(unpacked, label, source, imported=True)
        os.rename(unpacked, destination)
    except (OSError, tarfile.TarError, RejectedArchive) as exc:
        return None, [f"could not unpack: {exc}"]
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return target_id, []


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
#
# Kept beside the manifest rather than in it. The manifest is the provenance of
# a measurement -- what machine, what versions, what limits -- and editing it to
# add a nickname would make it a thing that has been edited.

LABEL_NAME = "vb-label.json"


def _write_label(run_dir: str, label: Optional[str], source: Optional[str],
                 imported: bool = False) -> None:
    existing = read_label(run_dir)
    data: Dict[str, Any] = dict(existing)
    if label is not None:
        data["label"] = label.strip()[:200]
    if source:
        data["source"] = source.strip()[:200]
    if imported:
        data["imported_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not data:
        return
    try:
        path = os.path.join(run_dir, LABEL_NAME)
        temporary = f"{path}.tmp"
        with open(temporary, "w") as fh:
            json.dump(data, fh, indent=1)
        os.replace(temporary, path)
    except OSError:
        pass


def read_label(run_dir: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(run_dir, LABEL_NAME)) as fh:
            found = json.load(fh)
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def set_label(run_dir: str, label: str, source: Optional[str] = None) -> None:
    _write_label(run_dir, label, source)


def clear_label(run_dir: str) -> None:
    try:
        os.unlink(os.path.join(run_dir, LABEL_NAME))
    except OSError:
        pass
