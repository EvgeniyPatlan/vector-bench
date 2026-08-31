"""Reading, validating and writing config/engines/*.yml.

What this makes possible is adding a *variant* of an engine family the harness
already drives -- another MySQL fork, another Postgres build -- because that is
genuinely just configuration. It cannot add a new architecture: that needs a
driver in harness/drivers/, an ann-benchmarks module and a Dockerfile, and the
validator says so by name rather than letting a config through that no driver
can serve.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

NAME_RE = re.compile(r"^[a-z][a-z0-9]{1,31}$")

REQUIRED_RUNTIME = ("driver", "ann_constructor", "port", "data_mount", "probe")
REQUIRED_TOP = ("name", "display_name", "image", "runtime")


def engines_dir(config_dir: str) -> str:
    return os.path.join(config_dir, "engines")


def engine_path(config_dir: str, name: str) -> Optional[str]:
    if not NAME_RE.match(name or ""):
        return None
    return os.path.join(engines_dir(config_dir), f"{name}.yml")


def available_drivers() -> List[str]:
    from harness.drivers.postgres import _driver_classes
    return sorted(_driver_classes())


def listing(config_dir: str) -> List[Dict[str, Any]]:
    from orchestrator import engines as registry_mod
    from orchestrator.config import load_engine

    out: List[Dict[str, Any]] = []
    for name, engine in registry_mod.registry(refresh=True).items():
        try:
            config = load_engine(name)
        except FileNotFoundError:
            config = {}
        images = config.get("image") or {}
        out.append({
            "name": name,
            "display_name": engine.display_name,
            "label": engine.label,
            "tag": engine.tag,
            "driver": engine.driver,
            "ann_constructor": engine.ann_constructor,
            "port": engine.port,
            "group": engine.group,
            "order": engine.order,
            "color": engine.color,
            "family": config.get("family"),
            "images": {"runtime": images.get("runtime", ""),
                       "bench": images.get("bench", "")},
            "capabilities": config.get("capabilities") or {},
        })
    return out


def read(config_dir: str, name: str) -> Optional[Dict[str, Any]]:
    path = engine_path(config_dir, name)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return {"name": name, "text": text, "parsed": {}, "errors": [str(exc)]}
    return {"name": name, "text": text, "parsed": parsed, "errors": []}


def validate(config_dir: str, name: str, text: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Return (errors, warnings, parsed)."""
    errors: List[str] = []
    warnings: List[str] = []

    if not NAME_RE.match(name or ""):
        errors.append("engine name must be lowercase letters and digits, "
                      "2-32 characters (it becomes a Docker container name)")

    try:
        parsed = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        return errors + [f"YAML error: {exc}"], warnings, {}

    if not isinstance(parsed, dict):
        return errors + ["an engine config must be a YAML mapping"], warnings, {}

    for key in REQUIRED_TOP:
        if key not in parsed:
            errors.append(f"missing required key: {key}")

    if parsed.get("name") and parsed["name"] != name:
        errors.append(f"name in file is {parsed['name']!r}, expected {name!r}")

    runtime = parsed.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be a mapping")
        return errors, warnings, parsed

    for key in REQUIRED_RUNTIME:
        if not runtime.get(key):
            errors.append(f"runtime.{key} is required")

    driver = runtime.get("driver")
    drivers = available_drivers()
    if driver and driver not in drivers:
        errors.append(
            f"runtime.driver {driver!r} does not exist. This form can add a "
            f"variant of a family the harness already drives ({', '.join(drivers)}); "
            f"a new architecture needs a driver in harness/drivers/, an "
            f"ann-benchmarks module and a Dockerfile. See docs/08-web-ui.md.")

    port = runtime.get("port")
    if port is not None and (not isinstance(port, int) or not 1 <= port <= 65535):
        errors.append("runtime.port must be an integer between 1 and 65535")

    probe = runtime.get("probe")
    if probe is not None and (not isinstance(probe, list) or not all(
            isinstance(a, str) for a in probe)):
        errors.append("runtime.probe must be a list of strings (argv)")

    errors.extend(_collision_errors(config_dir, name, parsed, runtime))

    images = parsed.get("image") or {}
    if isinstance(images, dict):
        for kind in ("runtime", "bench"):
            if not images.get(kind):
                errors.append(f"image.{kind} is required")
        others = _other_images(config_dir, name)
        for kind in ("runtime", "bench"):
            ref = images.get(kind)
            if ref and ref in others:
                errors.append(
                    f"image.{kind} {ref!r} is already used by "
                    f"{others[ref]}; two engines sharing an image tag would "
                    f"overwrite each other at build time")

    if not (parsed.get("sql") or parsed.get("runtime", {}).get("driver") in
            ("MongoDriver", "ValkeyDriver")):
        warnings.append("no sql block: SQL-family drivers read their dialect from it")

    return errors, warnings, parsed


def _collision_errors(config_dir: str, name: str, parsed: Dict[str, Any],
                      runtime: Dict[str, Any]) -> List[str]:
    """ann-benchmarks keys its result files on the algorithm name.

    Two engines sharing an ann_constructor therefore share a results tree, and
    the second silently reports the first's numbers. This is exactly why
    mariadb123 exists as a separate constructor rather than a retagged mariadb.
    """
    from orchestrator.config import load_engine

    errors: List[str] = []
    constructor = runtime.get("ann_constructor")
    if not constructor:
        return errors

    directory = engines_dir(config_dir)
    try:
        names = [f[:-4] for f in os.listdir(directory) if f.endswith(".yml")]
    except OSError:
        return errors

    for other in names:
        if other == name:
            continue
        try:
            theirs = (load_engine(other).get("runtime") or {}).get("ann_constructor")
        except (FileNotFoundError, Exception):  # noqa: BLE001
            continue
        if theirs and theirs == constructor:
            errors.append(
                f"runtime.ann_constructor {constructor!r} is already used by "
                f"{other}. ann-benchmarks keys result files on it, so both "
                f"engines would share one results tree and the second would "
                f"report the first's recall numbers.")
    return errors


def _other_images(config_dir: str, name: str) -> Dict[str, str]:
    from orchestrator.config import load_engine
    used: Dict[str, str] = {}
    try:
        names = [f[:-4] for f in os.listdir(engines_dir(config_dir)) if f.endswith(".yml")]
    except OSError:
        return used
    for other in names:
        if other == name:
            continue
        try:
            images = load_engine(other).get("image") or {}
        except Exception:  # noqa: BLE001
            continue
        for ref in (images.get("runtime"), images.get("bench")):
            if ref:
                used[ref] = other
    return used


def clone(config_dir: str, base: str, name: str) -> Tuple[Optional[str], List[str]]:
    """A copy of `base` renamed to `name`, as text, not written.

    Text rather than a re-serialised mapping: the engine configs carry more
    explanation than settings, and round-tripping through yaml.dump would throw
    all of it away.
    """
    if not NAME_RE.match(name or ""):
        return None, ["engine name must be lowercase letters and digits, 2-32 characters"]
    if engine_path(config_dir, name) and os.path.exists(engine_path(config_dir, name)):
        return None, [f"{name} already exists"]

    source = read(config_dir, base)
    if source is None:
        return None, [f"no such engine to copy: {base}"]

    parsed = source["parsed"]
    old_constructor = (parsed.get("runtime") or {}).get("ann_constructor", "")
    new_constructor = name[:1].upper() + name[1:]

    text = source["text"]
    replacements = [
        (f"\nname: {base}\n", f"\nname: {name}\n"),
        (f"vector-bench/{base}-runtime", f"vector-bench/{name}-runtime"),
        (f"vector-bench/{base}-bench", f"vector-bench/{name}-bench"),
    ]
    if old_constructor:
        replacements.append((f"ann_constructor: {old_constructor}",
                             f"ann_constructor: {new_constructor}"))
    for old, new in replacements:
        text = text.replace(old, new)

    banner = (
        f"# Copied from {base}. Change the source tag, the image names and the\n"
        f"# server flags to describe the new build; the driver stays the same,\n"
        f"# which is what makes this a config change rather than a code change.\n"
        f"#\n"
        f"# Build it with: ./run-benchmark.sh build --engines {name}\n"
    )
    return banner + text, []


def write(config_dir: str, name: str, text: str) -> Tuple[bool, List[str], List[str]]:
    errors, warnings, _parsed = validate(config_dir, name, text)
    if errors:
        return False, errors, warnings

    path = engine_path(config_dir, name)
    if path is None:
        return False, ["invalid engine name"], warnings

    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        os.replace(temporary, path)
    except OSError as exc:
        return False, [f"could not write {path}: {exc}"], warnings

    from webui.profiles import _match_owner
    _match_owner(directory, path)

    from orchestrator import engines as registry_mod
    registry_mod.registry(refresh=True)
    return True, [], warnings
