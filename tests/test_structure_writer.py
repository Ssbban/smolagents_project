"""write_structure: atoms the agent worked out, checked and written as extended XYZ with the cell.

Asked on the cluster for an NVE run with "any" structure, both agents built 64-atom silicon in memory and could not
save it: generated code cannot write files and no tool wrote a structure.
"""

import re
from pathlib import Path

import pytest

from tools.ipi_InputFile_generator import ipi_InputFile_generator
from tools.slurm_submit import slurm_submit
from tools.structure_writer import write_structure

A = 5.431  # diamond silicon, angstrom
BASIS = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0), (.25, .25, .25), (.25, .75, .75), (.75, .25, .75),
         (.75, .75, .25)]
SI64 = [[(i + x) * A, (j + y) * A, (k + z) * A] for i in range(2) for j in range(2) for k in range(2) for x, y, z in BASIS]
WATER = {"name": "water", "symbols": ["O", "H", "H"], "positions": [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
         "cell": [10, 10, 10], "pbc": False}


def test_a_crystal_is_written_as_extended_xyz_with_its_cell(session):
    result = write_structure(name="si64", symbols=["Si"] * 64, positions=SI64, cell=[2 * A] * 3)
    assert (result["formula"], result["atoms"]) == ("Si64", 64)
    assert result["shortest_distance"] == pytest.approx(A * 3 ** 0.5 / 4, abs=1e-4)  # the Si-Si bond
    lines = Path(result["structure_path"]).read_text().splitlines()
    assert lines[0] == "64" and len(lines) == 66 and lines[2].split()[0] == "Si"
    assert lines[1].startswith('Lattice="10.86200000 0.00000000 0.00000000 0.00000000 10.86200000')
    assert lines[1].endswith('Properties=species:S:1:pos:R:3 pbc="T T T"')
    assert session.find_artifact(result["structure_path"]).kind == "structure"


def test_the_written_file_is_what_the_ipi_input_generator_takes(session):
    path = write_structure(**WATER)["structure_path"]
    xml = ipi_InputFile_generator("03-NVT-SVR", path)
    assert xml["facts"]["structure"] == "water.xyz"
    assert [a.kind for a in session.artifacts] == ["structure", "structure", "ipi_input"]  # copied into the bundle


@pytest.mark.parametrize(
    "changes, fragment",
    [
        ({"symbols": ["O", "H", "Hx"]}, "unknown element symbols: ['Hx']"),
        ({"symbols": ["O", "H"]}, "positions must be one [x, y, z] in angstrom for each of the 2 symbols"),
        ({"positions": [[0, 0, 0], [0.757, 0.586, 0], [0, 0, "x"]]}, "positions must be one [x, y, z]"),
        ({"cell": [10, 10, 0]}, "the cell must have a positive volume"),
        ({"cell": [[10, 0, 0], [0, 0, 10], [0, 10, 0]]}, "three right-handed vectors"),  # i-PI: negative volume
        ({"cell": [10, 10, 0.4], "pbc": True}, "the cell is only 0.400 angstrom wide"),
        ({"symbols": [["O", "H"], "H", "H"]}, "unknown element symbols: [\"['O', 'H']\"]"),
        ({"cell": [10, 10]}, "cell must be three lengths"),
        ({"name": "../job"}, "name may only contain"),
        ({"positions": [[0, 0, 0], [0.1, 0, 0], [-0.757, 0.586, 0]]}, "atoms 0 (O) and 1 (H) are 0.100 angstrom apart"),
    ],
)
def test_bad_atoms_are_refused_and_nothing_is_written(session, changes, fragment):
    with pytest.raises(ValueError, match=re.escape(fragment)):
        write_structure(**{**WATER, **changes})
    assert session.artifacts == []


def test_overlaps_count_periodic_images_and_catch_fractional_coordinates(session):
    across = {"name": "pair", "symbols": ["Ar", "Ar"], "positions": [[0.1, 0, 0], [4.9, 0, 0]], "cell": [5, 5, 5]}
    with pytest.raises(ValueError, match="0.200 angstrom apart"):  # close through the cell boundary
        write_structure(**across)
    assert write_structure(**{**across, "pbc": False})["shortest_distance"] == pytest.approx(4.8)
    with pytest.raises(ValueError, match="fractional coordinates given as Cartesian"):
        write_structure(name="si8", symbols=["Si"] * 8, positions=[list(b) for b in BASIS], cell=[A] * 3)


def test_a_written_structure_is_never_a_job_script(session):
    assert slurm_submit(write_structure(**{**WATER, "name": "job"})["structure_path"])["status"] == "refused"


def test_a_name_given_with_its_extension_is_not_doubled(session):
    assert write_structure(**{**WATER, "name": "water.xyz"})["structure_path"].endswith("/water.xyz")


def test_large_structures_are_written_with_a_note_instead_of_the_pair_check(session, monkeypatch):
    monkeypatch.setattr("tools.structure_writer.MAX_CHECKED", 2)
    result = write_structure(**WATER)
    assert result["shortest_distance"] is None and result["notes"] == ["overlaps were not checked: more than 2 atoms"]
