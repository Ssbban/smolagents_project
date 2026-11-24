"""Look-up of i-PI input tags in the installed i-PI's own input classes.

Every i-PI input class keeps its tags in the class dictionaries `attribs`, `fields` and `dynamic`
(name -> (InputClass, {dtype, default, options, dimension, help})). Walking them from InputSimulation
gives the tag tree of exactly the i-PI version that will read the input; nothing is parsed or built.
"""

import contextlib
import copy
import functools
import io
import json
from typing import Any, Dict, List

from smolagents.tools import tool

MAX_DEPTH = 10  # deep enough for a thermostat inside a barostat inside dynamics inside motion
MAX_GROUPS = 6
MAX_PATHS = 5
COMMON_CONTEXTS = ("simulation/system/motion/dynamics/", "simulation/system/ensemble/", "simulation/system/initialize")


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return type(value).__name__


def _describe(path: str, kind: str, cls: Any, kw: Dict[str, Any]) -> Dict[str, Any]:
    dtype = kw.get("dtype")
    entry = {
        "path": path,
        "kind": kind,
        "help": " ".join(str(kw.get("help") or getattr(cls, "default_help", "") or "").split()),
        "type": getattr(dtype, "__name__", None if dtype is None else str(dtype)),
        "default": _plain(kw.get("default")),
        "options": [str(option) for option in kw.get("options") or []] or None,
        "dimension": None if kw.get("dimension") in (None, "undefined") else kw.get("dimension"),
    }
    if kind == "element":
        attribs = getattr(cls, "attribs", None) or {}
        entry["attributes"] = {name: spec[1].get("options") if len(spec) > 1 else None for name, spec in attribs.items()}
        entry["children"] = [name for group in ("fields", "dynamic") for name in getattr(cls, group, None) or {}]
    return entry


@functools.lru_cache(maxsize=1)
def _index() -> tuple:
    """(i-PI version, every element and attribute of the input tree as described by i-PI)."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import ipi
        from ipi.inputs.simulation import InputSimulation

    entries: List[Dict[str, Any]] = [_describe("simulation", "element", InputSimulation, {})]

    def walk(cls: Any, path: str, ancestors: frozenset, depth: int) -> None:
        for group in ("attribs", "fields", "dynamic"):
            for name, spec in (getattr(cls, group, None) or {}).items():
                sub_cls, kw = spec[0], spec[1] if len(spec) > 1 else {}
                if group == "attribs":
                    entries.append(_describe(f"{path}@{name}", "attribute", sub_cls, kw))
                    continue
                entries.append(_describe(f"{path}/{name}", "element", sub_cls, kw))
                if sub_cls not in ancestors and depth < MAX_DEPTH:  # a class may contain itself
                    walk(sub_cls, f"{path}/{name}", ancestors | {sub_cls}, depth + 1)

    walk(InputSimulation, "simulation", frozenset({InputSimulation}), 0)
    return str(getattr(ipi, "__version__", "unknown")), entries


def _name(path: str) -> str:
    return path.rsplit("@", 1)[1] if "@" in path else path.rsplit("/", 1)[-1]


def _rank(entry: Dict[str, Any]) -> tuple:
    """Places a molecular dynamics input uses first, rarer motions (scp, planetary, ...) later."""
    path = entry["path"]
    common = path.startswith(COMMON_CONTEXTS) or path.count("/") <= 2
    return (not common, path.count("/"), path)


@tool
def ipi_tag_help(tag: str) -> Dict[str, Any]:
    """
    Describe an i-PI input tag or attribute as the installed i-PI defines it: help text, type, default,
    allowed options, dimension, attributes (name -> allowed values) and child tags. A value whose
    dimension is set (temperature, time, pressure, ...) takes a units attribute; without one i-PI reads
    it in atomic units. Result: {"ipi_version", "tag", "matches": [{"paths", "kind", "help", "type",
    "default", "options", "dimension", "attributes", "children"}], "more"}. The same tag can mean
    different things in different places; each match lists where it occurs.

    Args:
        tag: A tag or attribute name such as "thermostat", "tau" or "nbeads", or a path such as
            "system/motion/dynamics/thermostat" or "system/initialize@nbeads".
    """
    version, entries = _index()
    key = str(tag).strip().strip("/").lstrip("@").removeprefix("./")
    if "/" in key or "@" in key:
        path = key if key.startswith("simulation") else f"simulation/{key}"
        hits = [entry for entry in entries if entry["path"] == path]
    else:
        hits = sorted((entry for entry in entries if _name(entry["path"]) == key), key=_rank)
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in hits:
        description = {k: v for k, v in entry.items() if k != "path"}
        group = groups.setdefault(json.dumps(description, sort_keys=True), {"paths": [], **description})
        group["paths"].append(entry["path"])
    matches = list(groups.values())
    for match in matches:
        match["paths"] = match["paths"][:MAX_PATHS]
    result = {"ipi_version": version, "tag": key, "matches": matches[:MAX_GROUPS], "more": max(0, len(matches) - MAX_GROUPS)}
    if not matches:
        result["note"] = "no such tag or attribute in this i-PI version"
    return copy.deepcopy(result)  # the lists inside belong to the cached index
