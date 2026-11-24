"""i-PI's own loader on the vendored templates and on the unit trap (runs i-PI in a child process, no network)."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.ipi_check import check_input, validation_copy

TEMPLATES = Path(__file__).resolve().parents[1] / "tools" / "ipi_templates"
SOURCE = json.loads((TEMPLATES / "SOURCE.json").read_text(encoding="utf-8"))
CATALOG = json.loads((TEMPLATES / "catalog.json").read_text(encoding="utf-8"))
LANGEVIN = (TEMPLATES / "02-NVT-Langevin" / "input.xml").read_text(encoding="utf-8")


def test_every_vendored_file_matches_its_recorded_checksum():
    assert set(SOURCE["files"]) == {"LICENSE"} | {f"{name}/input.xml" for name in CATALOG}
    assert SOURCE["commit"] in SOURCE["files"]["LICENSE"]["url"]
    for name, record in SOURCE["files"].items():
        assert hashlib.sha256((TEMPLATES / name).read_bytes()).hexdigest() == record["sha256"], name


@pytest.mark.parametrize("name", sorted(CATALOG))
def test_every_template_loads_unmodified(name):
    result = check_input((TEMPLATES / name / "input.xml").read_text(encoding="utf-8"))
    assert result["problems"] == []
    facts = result["facts"]
    assert facts["total_steps"] == 20000 and facts["timestep_fs"][0] == pytest.approx(0.5)
    assert facts["forcefields"] == [{"type": "ffsocket", "name": "driver", "mode": "unix", "address": "driver"}]
    expected = {"07-REMD-NVT": (5, 1), "08-REMD-NPT": (5, 1), "09-PINVT-for-dynamics": (1, 32)}.get(name, (1, 1))
    assert (facts["replicas"], facts["nbeads"]) == expected
    if name.startswith(("07", "08")):
        assert facts["temperature_K"] == pytest.approx([260, 280, 300, 320, 340])
    elif name != "00-NVE":
        assert facts["temperature_K"] == pytest.approx([300])
    has_barostat = name.startswith(("05", "06", "08"))
    assert all((p is not None) == has_barostat for p in facts["pressure_bar"])


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("<temperature units='kelvin'> 300 </temperature>", "<temperature> 300 </temperature>", "9.473e+07 K"),
        ("<timestep units='femtosecond'> 0.50 </timestep>", "<timestep> 0.50 </timestep>", "0.01209 fs"),
        ("<temperature units='kelvin'> 300 </temperature>", "<temperature units='kelvin'> 0.5 </temperature>",
         "0.5 K"),
    ],
)
def test_the_unit_trap_is_reported_with_the_value_ipi_understood(old, new, fragment):
    assert old in LANGEVIN
    problems = check_input(LANGEVIN.replace(old, new))["problems"]
    assert len(problems) == 1 and fragment in problems[0] and "atomic units" in problems[0]


def test_a_pressure_without_units_is_reported():
    text = (TEMPLATES / "05-NPT-isotropic-BZP" / "input.xml").read_text(encoding="utf-8")
    problems = check_input(text.replace("<pressure units='bar'> 1.0 </pressure>", "<pressure> 1.0 </pressure>"))
    assert len(problems["problems"]) == 1 and "2.942e+08 bar" in problems["problems"][0]


@pytest.mark.parametrize(
    "old, new, fragment",
    [
        ("<total_steps>", "<bogus>1</bogus><total_steps>", "i-PI: NameError: Tag name 'bogus' is not a recognized"),
        ("verbosity='high'", "verbosity='loud'", "loud is not a valid option"),
        ("mode='langevin'", "mode='langevinx'", "langevinx is not a valid option"),
        ("</simulation>", "<prng><seed>1</seed></prng></simulation>", "<prng> appears 2 times in <simulation>"),
    ],
)
def test_ipi_messages_are_returned(old, new, fragment):
    problems = check_input(LANGEVIN.replace(old, new))["problems"]
    assert any(fragment in problem for problem in problems), problems


def test_validation_copy_has_no_socket_and_no_structure_file():
    copy = ET.fromstring(validation_copy(ET.fromstring(LANGEVIN)))
    assert copy.find("ffsocket") is None
    assert copy.find("ffdirect").get("name") == "driver" and copy.findtext("ffdirect/pes") == "dummy"
    assert copy.findtext("system/initialize/file") == "placeholder.xyz"


def test_a_check_that_does_not_finish_is_a_problem_and_is_not_kept():
    text = LANGEVIN.replace("31415", "27182")
    assert "did not finish" in check_input(text, timeout=0.001)["problems"][0]
    assert check_input(text)["problems"] == []


def test_not_well_formed_xml_is_a_problem():
    assert "not well-formed" in check_input("<simulation>")["problems"][0]
