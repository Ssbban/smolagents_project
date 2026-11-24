"""Check an i-PI input with the installed i-PI before it is written or queued.

i-PI builds every force field while it loads an input, so it is given a validation copy: each force
field is replaced by i-PI's dummy potential and each structure file by a three-atom placeholder. The
copy is loaded in a child process (`python -I tools/ipi_check.py input.xml`) with a time-out; the
child imports only the standard library and ipi, and prints one line of JSON after MARKER.

check_input() returns i-PI's own messages as problems, the problems of a few physical sanity checks
(a value written without units is read in atomic units: "300" kelvin becomes about 9.5e7 K) and facts
read from the loaded simulation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MARKER = "@@IPI_CHECK@@ "
SOCKETS_DIR = "/tmp/ipi_"  # i-PI's default prefix of unix sockets; the ASE client always uses it
PLACEHOLDER_NAME = "placeholder.xyz"
# A water molecule in a 10 angstrom box: enough for i-PI to build every template's system.
PLACEHOLDER_XYZ = """3
# CELL(abcABC): 10.0 10.0 10.0 90.0 90.0 90.0 positions{angstrom} cell{angstrom}
O 0.0 0.0 0.0
H 0.757 0.586 0.0
H -0.757 0.586 0.0
"""
TEMPERATURE_K = (1.0, 10_000.0)
TIMESTEP_FS = (0.05, 10.0)
MAX_PRESSURE_BAR = 1e7  # 1 TPa; one atomic unit of pressure is about 2.9e8 bar
# Validation copy -> JSON answer of the child. Inputs that differ only in socket address or structure file
# (what the copy replaces) share one i-PI run.
_answers: dict[str, str] = {}


def forcefields(root: ET.Element) -> list[dict]:
    """The force fields declared in the input, with i-PI's defaults for a socket (mode inet, localhost)."""
    found = []
    for element in root:
        if isinstance(element.tag, str) and element.tag.startswith("ff"):
            socket = element.tag == "ffsocket"
            found.append({
                "type": element.tag,
                "name": element.get("name"),
                "mode": element.get("mode", "inet") if socket else None,
                "address": (element.findtext("address") or "localhost").strip() if socket else None,
            })  # fmt: skip
    return found


def structure_files(root: ET.Element) -> list[str]:
    """Text of every <file> of an <initialize> block (the structure i-PI starts from)."""
    return [(file.text or "").strip() for init in root.iter("initialize") for file in init.findall("file")]


def links(xml_text: str) -> dict:
    """What a client must agree on with an input: {"socket": unix socket address or None, "structure": path or None}."""
    root = ET.fromstring(xml_text)
    sockets = [ff["address"] for ff in forcefields(root) if ff["type"] == "ffsocket" and ff["mode"] == "unix"]
    files = structure_files(root)
    return {"socket": sockets[0] if len(sockets) == 1 else None, "structure": files[0] if files else None}


def validation_copy(root: ET.Element) -> str:
    """The input with dummy force fields and the placeholder structure: loading it runs nothing and reads nothing."""
    copy = ET.fromstring(ET.tostring(root))
    for index, element in enumerate(list(copy)):
        if isinstance(element.tag, str) and element.tag.startswith("ff"):
            dummy = ET.Element("ffdirect", name=element.get("name", "dummy"))
            ET.SubElement(dummy, "pes").text = "dummy"
            copy.remove(element)
            copy.insert(index, dummy)
    for init in copy.iter("initialize"):
        for file in init.findall("file"):
            file.attrib.clear()
            file.attrib.update(mode="xyz", units="angstrom", cell_units="angstrom")
            file.text = PLACEHOLDER_NAME
    return ET.tostring(copy, encoding="unicode")


def _run_child(text: str, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="ipi-check-") as tmp:
        Path(tmp, "input.xml").write_text(text, encoding="utf-8")
        Path(tmp, PLACEHOLDER_NAME).write_text(PLACEHOLDER_XYZ, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(Path(__file__).resolve()), "input.xml"],
                cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return {"failure": f"the i-PI check did not finish in {timeout:g} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    tail = proc.stderr.strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
    return {"failure": f"the i-PI check ended without an answer: {tail[0][:300]}"}


def _range_problems(facts: dict) -> list[str]:
    problems = []
    for temperature in facts.get("temperature_K", []):
        if temperature is not None and not TEMPERATURE_K[0] <= temperature <= TEMPERATURE_K[1]:
            problems.append(f"i-PI reads the temperature as {temperature:.4g} K; a temperature without "
                            "units='kelvin' is read in atomic units (300 becomes about 9.5e7 K)")
    for timestep in facts.get("timestep_fs", []):
        if timestep is not None and not TIMESTEP_FS[0] <= timestep <= TIMESTEP_FS[1]:
            problems.append(f"i-PI reads the time step as {timestep:.4g} fs, outside {TIMESTEP_FS[0]}-{TIMESTEP_FS[1]} "
                            "fs; a time step without units='femtosecond' is read in atomic units (0.5 is 0.012 fs)")
    for pressure in facts.get("pressure_bar", []):
        if pressure is not None and abs(pressure) > MAX_PRESSURE_BAR:
            problems.append(f"i-PI reads the pressure as {pressure:.4g} bar; a pressure without units='bar' "
                            "is read in atomic units (1 becomes about 2.9e8 bar)")
    return problems


def check_input(xml_text: str, timeout: float = 60) -> dict:
    """{"problems": [...], "facts": {...}} for an i-PI input; problems are i-PI's messages and range checks.

    Facts: temperature_K, timestep_fs, pressure_bar (one entry per replica, None when not set),
    nbeads, replicas, total_steps, forcefields (type, name, mode, address as written in the input).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        return {"problems": [f"the input is not well-formed XML: {error}"], "facts": {}}
    copy = validation_copy(root)
    if copy not in _answers:
        answer = _run_child(copy, timeout)
        if "failure" in answer:  # not kept: the next call tries again
            return {"problems": [answer["failure"]], "facts": {"forcefields": forcefields(root)}}
        _answers[copy] = json.dumps(answer)
    answer = json.loads(_answers[copy])  # a fresh copy each time: callers may change what they get
    facts = {**(answer.get("facts") or {}), "forcefields": forcefields(root)}
    problems = answer.get("repeated") or []
    problems += [f"i-PI: {answer['error']}"] if answer.get("error") else _range_problems(facts)
    return {"problems": problems, "facts": facts}


# ---- child process: standard library and ipi only ----


def _system_facts(system, units) -> dict:
    def user(kind: str, unit: str, value: float) -> float:
        return float(units.unit_to_user(kind, unit, float(value)))

    temperature = float(system.ensemble.temp)
    timestep = getattr(system.motion, "dt", None)
    barostat = getattr(system.motion, "enstype", None) in ("npt", "scnpt")
    return {
        "temperature_K": user("temperature", "kelvin", temperature) if temperature >= 0 else None,
        "timestep_fs": None if timestep is None else user("time", "femtosecond", timestep),
        "pressure_bar": user("pressure", "bar", system.ensemble.pext) if barostat else None,
        "nbeads": int(system.beads.nbeads),
    }


def _repeated(node, obj, path: str) -> list[str]:
    """Tags written twice where i-PI keeps only one (it silently reads the last)."""
    names = [name for name, _ in node.fields if name != "_text"]
    found = [f"<{name}> appears {names.count(name)} times in <{path}>; i-PI reads only the last one"
             for name in dict.fromkeys(names) if names.count(name) > 1 and name in obj.instancefields]
    for name, child in node.fields:
        if name in obj.instancefields:
            found += _repeated(child, getattr(obj, name), f"{path}/{name}")
    dynamic = [(name, child) for name, child in node.fields if name in obj.dynamic]
    for (name, child), (_, child_obj) in zip(dynamic, obj.extra):
        found += _repeated(child, child_obj, f"{path}/{name}")
    return found


def _load(path: str) -> dict:
    import contextlib
    import io

    result: dict = {}
    text = Path(path).read_text(encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            from ipi.engine.simulation import Simulation
            from ipi.inputs.simulation import InputSimulation
            from ipi.utils import units
            from ipi.utils.io.inputs.io_xml import xml_parse_string

            node = xml_parse_string(text).fields[0][1]
            parsed = InputSimulation()
            parsed.parse(node)
            result["repeated"] = _repeated(node, parsed, "simulation")
            sim = Simulation.load_from_xml(text, read_only=True)
            systems = [_system_facts(system, units) for system in sim.syslist]
            result["facts"] = {
                "temperature_K": [s["temperature_K"] for s in systems],
                "timestep_fs": [s["timestep_fs"] for s in systems],
                "pressure_bar": [s["pressure_bar"] for s in systems],
                "nbeads": max((s["nbeads"] for s in systems), default=1),
                "replicas": len(systems),
                "total_steps": int(sim.tsteps),
            }
        except BaseException as error:  # i-PI reports bad input with any exception type, also SystemExit
            lines = [line for line in str(error).splitlines() if line.strip()]
            result["error"] = f"{type(error).__name__}: {lines[-1] if lines else ''}".strip()[:600]
    return result


if __name__ == "__main__":
    sys.stdout.write(MARKER + json.dumps(_load(sys.argv[1])) + "\n")
    sys.stdout.flush()
    os._exit(0)  # i-PI may leave threads behind; the answer is written
