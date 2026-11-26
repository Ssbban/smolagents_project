"""write_structure: a structure file built from atoms the agent computed, for a person who has none.

Generated code cannot write files, so a structure the agent works out (a crystal, a molecule in a box, a liquid
box) reaches the disk only through this tool, which checks the atoms and writes extended XYZ with the cell.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from smolagents.tools import tool

from tools.session import get_session
from tools.specs import _require_safe_name, check, raise_problems

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb "
    "Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au "
    "Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv "
    "Ts Og".split()
)
MIN_DISTANCE = 0.5  # angstrom; the shortest bond, in H2, is 0.74
MAX_CHECKED = 20_000  # atoms; the overlap check compares every pair, about 1 minute at this size
PAIRS_PER_CHUNK = 2_000_000  # pairs of the distance matrix held at a time (about 50 MB per temporary)
CELL_MESSAGE = "cell must be three lengths [a, b, c] or three vectors [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]"


def _floats(value: Any, message: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not np.isfinite(array).all():
        raise ValueError(message)
    return array


def _cell(cell: Any, pbc: bool) -> np.ndarray:
    """3x3 cell (rows are the vectors) from three lengths or three vectors, in angstrom."""
    box = _floats(cell, f"{CELL_MESSAGE} in angstrom; got {cell!r}")
    if box.shape not in ((3,), (3, 3)):
        raise ValueError(f"{CELL_MESSAGE} in angstrom; got {cell!r}")
    box = np.diag(box) if box.shape == (3,) else box
    volume = np.linalg.det(box)
    if volume < 1e-6:  # i-PI divides by the volume without abs: a left-handed cell flips the sign of the pressure
        raise ValueError("the cell must have a positive volume: three right-handed vectors (swap two of them if the "
                         "volume came out negative), and around a molecule a box larger than it")
    widths = volume / np.linalg.norm(np.cross(box[[1, 2, 0]], box[[2, 0, 1]]), axis=1)
    if pbc and widths.min() < 2 * MIN_DISTANCE:
        raise ValueError(f"the cell is only {widths.min():.3f} angstrom wide: every atom would overlap its own "
                         "periodic images")
    return box


def _positions(positions: Any, count: int) -> np.ndarray:
    message = f"positions must be one [x, y, z] in angstrom for each of the {count} symbols"
    xyz = _floats(positions, message)
    if xyz.shape != (count, 3):
        raise ValueError(message)
    return xyz


def shortest_distance(xyz: np.ndarray, box: np.ndarray, pbc: bool) -> tuple[float, int, int]:
    """(distance, i, j) of the closest pair of atoms. Periodic images by the minimum-image convention, exact for
    distances below half the narrowest cell width (the cell is at least twice MIN_DISTANCE wide)."""
    inverse = np.linalg.inv(box)
    rows = max(1, PAIRS_PER_CHUNK // len(xyz))
    best = (float("inf"), -1, -1)
    for start in range(0, len(xyz), rows):
        delta = xyz[start:start + rows, None, :] - xyz[None, :, :]
        if pbc:
            fractional = delta @ inverse
            delta = (fractional - np.round(fractional)) @ box
        distance = np.linalg.norm(delta, axis=-1)
        own = np.arange(len(distance))
        distance[own, own + start] = np.inf  # an atom and itself
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] < best[0]:
            best = (float(distance[i, j]), int(start + i), int(j))
    return best


def render_extxyz(symbols: List[str], xyz: np.ndarray, box: np.ndarray, pbc: bool) -> str:
    flag = "T" if pbc else "F"
    lattice = " ".join(f"{value:.8f}" for value in box.flatten())
    lines = [str(len(symbols)), f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="{flag} {flag} {flag}"']
    lines += [f"{symbol} {x:.8f} {y:.8f} {z:.8f}" for symbol, (x, y, z) in zip(symbols, xyz)]
    return "\n".join(lines) + "\n"


def formula(symbols: List[str]) -> str:
    return "".join(f"{element}{count if count > 1 else ''}" for element, count in Counter(symbols).items())


@tool
def write_structure(
    name: str, symbols: List[str], positions: List[List[float]], cell: List[Any], pbc: bool = True
) -> Dict[str, Any]:
    """
    Write a structure the person has no file for (a crystal, a molecule in a box, a liquid box) as an extended XYZ
    file with its cell, into a new bundle. Compute the atoms in your code and pass the lists. The atoms are checked:
    known element symbols, one position per symbol, a right-handed cell with volume, and no two atoms closer than
    0.5 angstrom (periodic images included when pbc is true). Invalid input raises one error listing every problem.
    Result: {"structure_path", "bundle", "formula", "atoms", "cell", "shortest_distance", "notes"}; pass
    structure_path to ipi_InputFile_generator as structure_path.

    Args:
        name: File name without extension: letters, digits, '.', '_' and '-', e.g. "si64".
        symbols: Element symbol of each atom, e.g. ["O", "H", "H"].
        positions: Cartesian position of each atom in angstrom (not fractional), e.g. [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]].
        cell: Cell in angstrom: three lengths [a, b, c] of a rectangular box, or three right-handed vectors [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]. Around a molecule, leave several angstrom of space on each side.
        pbc: Whether the cell is periodic: true for crystals and liquids.
    """
    problems: List[str] = []
    if isinstance(name, str):
        name = name.removesuffix(".xyz")
    check(problems, _require_safe_name, "name", name)
    if not isinstance(symbols, list) or not symbols:
        problems.append("symbols must be a non-empty list of element symbols, e.g. ['O', 'H', 'H']")
        symbols = []
    unknown = sorted({str(symbol) for symbol in symbols if str(symbol) not in ELEMENTS})
    if unknown:
        problems.append(f"unknown element symbols: {unknown} (symbols are case-sensitive, e.g. 'Si', not 'SI')")
    if not isinstance(pbc, bool):
        problems.append("pbc must be true or false")
    xyz: Optional[np.ndarray] = check(problems, _positions, positions, len(symbols)) if symbols else None
    box: Optional[np.ndarray] = check(problems, _cell, cell, pbc is not False)
    distance, notes = None, []
    if not problems and len(symbols) > MAX_CHECKED:
        notes.append(f"overlaps were not checked: more than {MAX_CHECKED} atoms")
    elif not problems and len(symbols) > 1:
        distance, i, j = shortest_distance(xyz, box, pbc)
        if distance < MIN_DISTANCE:
            problems.append(f"atoms {i} ({symbols[i]}) and {j} ({symbols[j]}) are {distance:.3f} angstrom apart, "
                            f"less than {MIN_DISTANCE}: overlapping atoms, a cell too small for them, or fractional "
                            "coordinates given as Cartesian")
    raise_problems("write_structure", problems)

    session = get_session()
    bundle = session.new_bundle("structure")
    artifact = session.write_artifact(bundle, f"{name}.xyz", render_extxyz(symbols, xyz, box, pbc), "structure")
    return {
        "structure_path": artifact.path,
        "bundle": artifact.bundle,
        "formula": formula(symbols),
        "atoms": len(symbols),
        "cell": box.round(6).tolist(),
        "shortest_distance": None if distance is None else round(distance, 4),
        "notes": notes,
    }
