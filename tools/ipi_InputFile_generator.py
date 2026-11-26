"""i-PI input files from the templates vendored in tools/ipi_templates (pinned upstream commit, see SOURCE.json)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from smolagents.tools import tool

from tools.ipi_check import check_input, forcefields
from tools.session import get_session
from tools.specs import SAFE_NAME_RE, IPIInputSpec, _require_safe_path, check, raise_problems

TEMPLATE_DIR = Path(__file__).with_name("ipi_templates")
CATALOG: Dict[str, Dict[str, str]] = json.loads((TEMPLATE_DIR / "catalog.json").read_text(encoding="utf-8"))
SOURCE: Dict[str, Any] = json.loads((TEMPLATE_DIR / "SOURCE.json").read_text(encoding="utf-8"))
# Text formats ASE recognises by file name (i-PI's mode='ase' and the client both read with ase.io.read).
STRUCTURE_SUFFIXES = {".xyz", ".extxyz", ".cif", ".pdb", ".vasp", ".poscar", ".xsf", ".gen", ".res"}
STRUCTURE_NAMES = {"POSCAR", "CONTCAR"}


def _select_template(template_type: str) -> str:
    return (TEMPLATE_DIR / template_type / "input.xml").read_text(encoding="utf-8")


def _tool_owned(path: str, tag: str) -> bool:
    """The socket, the structure file and the force field reference are written by the tools, never edited."""
    return tag.startswith("ff") or path.endswith(("/initialize/file", "/forces/force"))


def _element_paths(root: ET.Element) -> List[Dict[str, Any]]:
    """Every editable element of a template: exact xpath, tag, attributes and text."""
    paths: List[Dict[str, Any]] = []

    def visit(element: ET.Element, path: str) -> None:
        if path != ".":
            text = (element.text or "").strip()
            paths.append({"xpath": path, "tag": element.tag, "attrs": dict(element.attrib), "text": text or None})
        totals: Dict[str, int] = {}
        for child in element:
            totals[child.tag] = totals.get(child.tag, 0) + 1
        counts: Dict[str, int] = {}
        for child in element:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            child_path = f"{path}/{child.tag}"
            if totals[child.tag] > 1:
                child_path = f"{child_path}[{counts[child.tag]}]"
            if not _tool_owned(child_path, child.tag):
                visit(child, child_path)

    visit(root, ".")
    return paths


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dictionary.")
    return value


def _require_xml_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value) or not value[0].isalpha():
        raise ValueError(f"{name} must be a valid XML tag or attribute name.")
    return value


def _normalize_attrs(name: str, attrs: Any) -> Dict[str, str]:
    if attrs is None:
        return {}
    return {_require_xml_name(f"{name} key", key): str(value) for key, value in _require_mapping(name, attrs).items()}


def _add_child_element(parent: ET.Element, block: Mapping[str, Any]) -> None:
    tag = _require_xml_name("tag", block.get("tag"))
    if tag.startswith("ff"):
        raise ValueError(f"<{tag}> cannot be added: the force field is the template's socket, set by the tool")
    fields = block.get("fields") or []
    if not isinstance(fields, list):
        raise ValueError("fields must be a list.")
    child = ET.SubElement(parent, tag, _normalize_attrs("attrs", block.get("attrs")))
    if block.get("text") is not None:
        child.text = str(block["text"])
    for index, subfield in enumerate(fields):
        _add_child_element(child, _require_mapping(f"field {index}", subfield))


def _apply(root: ET.Element, editable: set, name: str, operation: Any) -> None:
    """One named operation: an update of an editable path, or a block appended under one (or under '.')."""
    operation = _require_mapping(name, operation)
    hint = "use an xpath that ipi_template_schema(template_type) lists for this template"
    if "xpath" in operation and "parent_xpath" in operation:
        raise ValueError(f"{name} must be either an xpath update or an append block, not both.")
    if "xpath" in operation:
        xpath = str(operation["xpath"])
        if xpath not in editable:
            raise ValueError(f"{name}.xpath {xpath!r} is not an editable path of this template; {hint}")
        attrs = _normalize_attrs(f"{name}.attrs", operation.get("attrs"))
        if not attrs and "text" not in operation:
            raise ValueError(f"{name} must include text or attrs.")
        element = root.find(xpath)
        element.attrib.update(attrs)
        if "text" in operation:
            element.text = str(operation["text"])
    elif "parent_xpath" in operation and "tag" in operation:
        parent_xpath = str(operation["parent_xpath"])
        if parent_xpath != "." and parent_xpath not in editable:
            raise ValueError(f"{name}.parent_xpath {parent_xpath!r} is not '.' or an editable path; {hint}")
        try:
            _add_child_element(root if parent_xpath == "." else root.find(parent_xpath), operation)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
    else:
        raise ValueError(f"{name} must include either xpath for an update or parent_xpath plus tag for an append block.")


def _structure(path: Any, on_cluster: bool) -> tuple:
    """(text, name) of a structure file readable here, or (None, path) for an absolute path on the cluster when the
    cluster's files are not visible from here."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("structure_path is required: the initial structure i-PI and the client start from")
    name = PurePosixPath(path.strip().replace("\\", "/")).name
    if PurePosixPath(name).suffix.lower() not in STRUCTURE_SUFFIXES and name not in STRUCTURE_NAMES:
        raise ValueError(f"structure_path {path!r} is not a structure file ASE recognises by its name: use one of "
                         f"{sorted(STRUCTURE_SUFFIXES)} or {sorted(STRUCTURE_NAMES)}")
    local = Path(path.strip()).expanduser()
    if local.is_file():
        if not SAFE_NAME_RE.fullmatch(local.name):
            raise ValueError(f"structure file name {local.name!r} may only contain letters, numbers, '.', '_' and '-'")
        try:
            return local.read_text(encoding="utf-8"), local.name
        except UnicodeDecodeError:
            raise ValueError(f"structure_path {path!r} is not a text structure file (xyz, extxyz, cif, ...)") from None
    if not PurePosixPath(path).is_absolute():
        raise ValueError(f"structure_path {path!r} is not a file on this machine; give a file readable here (it is "
                         "copied into the job's bundle) or its absolute path on the cluster (i-PI does not expand '~'). "
                         "Without a file, build the structure and write it with write_structure")
    if on_cluster:  # the agent runs on a login node, which sees the shared file systems
        raise ValueError(f"structure_path {path!r} does not exist here; give a file this login node can read (copy "
                         "one that only compute nodes see to the home or project directory), or build the structure "
                         "and write it with write_structure")
    return None, _require_safe_path("structure_path", path)


def _cell_problems(text: Optional[str], name: str, root: ET.Element) -> List[str]:
    """i-PI cannot start without a cell ("Initializer could not initialize the cell"): an XYZ file needs an extended
    XYZ Lattice=, unless the input gives a <cell> in <initialize>."""
    if text is None or PurePosixPath(name).suffix.lower() not in (".xyz", ".extxyz"):
        return []
    comment = text.split("\n", 2)[1] if text.count("\n") >= 2 else ""
    if "Lattice=" in comment or root.find(".//initialize/cell") is not None:
        return []
    return [f"structure {name!r} has no cell, and i-PI cannot start without one: give the file an extended XYZ "
            'Lattice="ax ay az bx by bz cx cy cz" on its second line, or append under ./system/initialize '
            "{\"tag\": \"cell\", \"attrs\": {\"mode\": \"manual\", \"units\": \"angstrom\"}, "
            "\"text\": \"[ax, ay, az, bx, by, bz, cx, cy, cz]\"} (a box large enough around a molecule)"]


def socket_address(bundle: str) -> str:
    """Unix socket name of a bundle: unique per session and bundle, so jobs on one node never share a socket."""
    bundle_path = PurePosixPath(bundle)
    return f"{bundle_path.parent.name}-{bundle_path.name}"


@tool
def ipi_template_schema(template_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Without an argument: the catalogue of i-PI templates (name, what it simulates, when to use it).
    With a template name: the paths of that template that ipi_InputFile_generator accepts in
    modifications, as {"template_type", "source", "paths": [{"xpath", "tag", "attrs", "text"}],
    "set_by_tool"}. Only these xpath values can be edited; the socket and the structure file are set
    by the tools and are not listed.

    Args:
        template_type: Template name from the catalogue; leave it out to get the catalogue.
    """
    if template_type is None:
        return {"templates": [{"name": name, **entry} for name, entry in CATALOG.items()]}
    spec = IPIInputSpec(template_type=template_type).validate(templates=CATALOG)
    return {
        "template_type": spec.template_type,
        "source": f"{SOURCE['repository']} at commit {SOURCE['commit']}",
        "paths": _element_paths(ET.fromstring(_select_template(spec.template_type))),
        "set_by_tool": "the socket (<ffsocket>), the structure file (<initialize><file>), the force reference",
    }


@tool
def ipi_InputFile_generator(
    template_type: str,
    structure_path: str,
    modifications: Optional[Dict[str, Any]] = None,
    simulation_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write an i-PI input.xml into a new job bundle, from a template plus exact edits.

    The input is loaded by the installed i-PI before it is written: values i-PI cannot read, unknown
    tags, tags written twice and implausible temperatures, time steps or pressures (a value without
    units is read in atomic units) are reported with the value i-PI understood. All problems come in
    one error; fix them and call again.
    Result: {"script_path", "bundle", "script_content", "facts"}; facts = what i-PI read: temperature_K,
    timestep_fs, pressure_bar (one entry per replica), nbeads, replicas, total_steps, forcefields,
    structure. Pass script_path to ipi_runase_generator and ipi_submit_generator.

    Args:
        template_type: Template name; ipi_template_schema() lists them.
        structure_path: Initial structure, read by i-PI and the client with ASE (extended XYZ with Lattice=... for a
            periodic cell). A file readable here is copied into the bundle, such as the structure_path that
            write_structure returns; otherwise the absolute path on the cluster.
        modifications: Operations, as a dict of name -> operation or as a list. An update is {"xpath": <path from
            ipi_template_schema>, "text": ..., "attrs": {...}} and keeps the units attribute unless you change it; an
            append block is {"parent_xpath": "." or a listed path, "tag": ..., "attrs": {...}, "text": ...,
            "fields": [blocks]}.
        simulation_attrs: Attributes of the root <simulation> tag: verbosity, threading, mode, safe_stride, floatformat.
    """
    session = get_session()
    if isinstance(modifications, list):  # operation names only label error messages
        modifications = {f"operation {number}": operation for number, operation in enumerate(modifications, 1)}
    spec = IPIInputSpec(template_type=template_type, modifications=modifications or {},
                        simulation_attrs=simulation_attrs or {})  # fmt: skip
    problems = spec.problems(templates=CATALOG)
    if not problems:
        root = ET.fromstring(_select_template(spec.template_type))
        editable = {entry["xpath"] for entry in _element_paths(root)}
        for name, operation in spec.modifications.items():
            check(problems, _apply, root, editable, str(name), operation)
        for attr, value in spec.simulation_attrs.items():
            root.set(str(attr), str(value))
    on_cluster = bool(session.cluster and session.cluster.get("available"))
    structure = check(problems, _structure, structure_path, on_cluster)
    if not problems:
        problems += _cell_problems(*structure, root)
    if not problems:
        result = check_input(ET.tostring(root, encoding="unicode"))
        problems += result["problems"]
    raise_problems("ipi_InputFile_generator", problems)

    bundle = session.new_bundle("ipi")
    text, name = structure
    if text is not None:
        name = PurePosixPath(session.write_artifact(bundle, name, text, "structure").path).name
    address = socket_address(Path(bundle).as_posix())
    for socket in root.iter("ffsocket"):
        socket.find("address").text = address
    for init in root.iter("initialize"):
        for file in init.findall("file"):
            file.text = name
    xml_content = ET.tostring(root, encoding="unicode")
    artifact = session.write_artifact(bundle, "input.xml", xml_content, "ipi_input")
    facts = {**result["facts"], "forcefields": forcefields(root), "structure": name}
    return {"script_path": artifact.path, "bundle": artifact.bundle, "script_content": xml_content, "facts": facts}
