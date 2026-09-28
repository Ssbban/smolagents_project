import os
import re
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional
import xml.etree.ElementTree as ET

import requests
from smolagents.tools import tool

from tools.specs import IPIInputSpec, IPI_TEMPLATE_URLS, WorkflowArtifact


IPI_TEMPLATE_BASE_URL = (
    "https://raw.githubusercontent.com/venkatkapil24/MLIPs-with-iPI/refs/heads/main"
)
XML_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _template_url(template_type: str) -> str:
    if template_type not in IPI_TEMPLATE_URLS:
        raise ValueError(
            f"Unknown template type: {template_type}. Available options: {list(IPI_TEMPLATE_URLS.keys())}"
        )
    return f"{IPI_TEMPLATE_BASE_URL}/{IPI_TEMPLATE_URLS[template_type]}"


def _select_template(template_type: str) -> str:
    response = requests.get(_template_url(template_type), timeout=30)
    response.raise_for_status()
    return response.text


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a dictionary.")
    return value


def _require_xml_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or not XML_NAME_RE.fullmatch(value):
        raise ValueError(f"{name} must be a valid XML tag or attribute name.")
    return value


def _normalize_attrs(name: str, attrs: Any) -> Dict[str, str]:
    if attrs is None:
        return {}
    attr_map = _require_mapping(name, attrs)
    return {
        _require_xml_name(f"{name} key", key): str(value)
        for key, value in attr_map.items()
    }


def _findall(root: ET.Element, xpath: str) -> List[ET.Element]:
    if xpath == ".":
        return [root]
    try:
        matches = root.findall(xpath)
    except SyntaxError as error:
        raise ValueError(f"Invalid XPath '{xpath}': {error}") from error
    if xpath == ".//simulation" and root.tag == "simulation" and root not in matches:
        matches = [root] + matches
    return matches


def _single_match(root: ET.Element, xpath: str, *, target: str) -> ET.Element:
    matches = _findall(root, xpath)
    if not matches:
        raise ValueError(f"{target} XPath did not match the selected template: {xpath}")
    if len(matches) > 1:
        raise ValueError(
            f"{target} XPath matched {len(matches)} elements. Use a more specific XPath: {xpath}"
        )
    return matches[0]


def _add_child_element(parent: ET.Element, field_config: Mapping[str, Any]) -> None:
    tag = _require_xml_name("field tag", field_config.get("tag"))
    attrs = _normalize_attrs("field attrs", field_config.get("attrs"))
    text = field_config.get("text")
    fields = field_config.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("field fields must be a list.")

    child = ET.SubElement(parent, tag)
    for attr_name, attr_value in attrs.items():
        child.set(attr_name, attr_value)
    if text is not None:
        child.text = str(text)
    for index, subfield in enumerate(fields):
        _add_child_element(child, _require_mapping(f"field {index}", subfield))


def _handle_xpath_update(
    root: ET.Element, operation_name: str, operation: Mapping[str, Any]
) -> None:
    xpath = operation.get("xpath")
    if not isinstance(xpath, str) or not xpath.strip():
        raise ValueError(f"{operation_name}.xpath is required.")
    attrs = _normalize_attrs(f"{operation_name}.attrs", operation.get("attrs"))
    has_text = "text" in operation
    if not attrs and not has_text:
        raise ValueError(f"{operation_name} must include text or attrs.")

    update_all = bool(operation.get("all", False))
    matches = _findall(root, xpath)
    if not matches:
        raise ValueError(f"{operation_name}.xpath did not match the selected template: {xpath}")
    if len(matches) > 1 and not update_all:
        raise ValueError(
            f"{operation_name}.xpath matched {len(matches)} elements. Set all=True only when every match should be updated."
        )

    for element in matches:
        for attr_name, attr_value in attrs.items():
            element.set(attr_name, attr_value)
        if has_text:
            element.text = str(operation["text"])


def _handle_append_block(
    root: ET.Element, operation_name: str, operation: Mapping[str, Any]
) -> None:
    parent_xpath = operation.get("parent_xpath")
    if not isinstance(parent_xpath, str) or not parent_xpath.strip():
        raise ValueError(f"{operation_name}.parent_xpath is required.")
    parent = _single_match(root, parent_xpath, target=f"{operation_name}.parent_xpath")

    block = {
        "tag": operation.get("tag"),
        "attrs": operation.get("attrs", {}),
        "text": operation.get("text"),
        "fields": operation.get("fields", []),
    }
    _add_child_element(parent, block)


def _modify_xml_template(xml_content: str, parameters: Mapping[str, Any]) -> str:
    root = ET.fromstring(xml_content)

    for operation_name, raw_operation in parameters.items():
        operation = _require_mapping(str(operation_name), raw_operation)
        has_xpath = "xpath" in operation
        has_block = "parent_xpath" in operation and "tag" in operation
        if has_xpath and has_block:
            raise ValueError(
                f"{operation_name} must be either an XPath update or an append block, not both."
            )
        if has_xpath:
            _handle_xpath_update(root, str(operation_name), operation)
        elif has_block:
            _handle_append_block(root, str(operation_name), operation)
        else:
            raise ValueError(
                f"{operation_name} must include either xpath for an exact update or parent_xpath plus tag for an explicit append block."
            )

    return ET.tostring(root, encoding="unicode")


def _element_paths(root: ET.Element) -> List[Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []

    def visit(element: ET.Element, path: str) -> None:
        text = (element.text or "").strip()
        paths.append(
            {
                "xpath": path,
                "tag": element.tag,
                "attrs": dict(element.attrib),
                "text": text if text else None,
            }
        )
        counts: Dict[str, int] = {}
        totals: Dict[str, int] = {}
        for child in list(element):
            totals[child.tag] = totals.get(child.tag, 0) + 1
        for child in list(element):
            counts[child.tag] = counts.get(child.tag, 0) + 1
            if path == ".":
                child_path = f"./{child.tag}"
            else:
                child_path = f"{path}/{child.tag}"
            if totals[child.tag] > 1:
                child_path = f"{child_path}[{counts[child.tag]}]"
            visit(child, child_path)

    visit(root, ".")
    return paths


@tool
def ipi_template_schema(template_type: str) -> Dict[str, Any]:
    """
    Fetch an i-PI template and list exact XML paths that can be updated.

    Args:
        template_type: Type of i-PI template to inspect.

    Returns:
        Dictionary containing the source URL, root tag and exact element paths.
    """
    spec = IPIInputSpec(template_type=template_type).validate()
    xml_content = _select_template(spec.template_type)
    root = ET.fromstring(xml_content)
    return {
        "template_type": spec.template_type,
        "source_url": _template_url(spec.template_type),
        "root_tag": root.tag,
        "paths": _element_paths(root),
    }


@tool
def ipi_InputFile_generator(
    template_type: str,
    modifications: Optional[Dict[str, Any]] = None,
    simulation_attrs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate an i-PI input file from a template using exact XML operations.

    Args:
        template_type: Type of simulation template to use as base.
        modifications: Named XML operations. Each operation must either include
            xpath with text and/or attrs for exact updates, or parent_xpath plus
            tag, attrs, text and fields for explicit append blocks.
        simulation_attrs: Optional validated attributes for the root simulation tag.

    Returns:
        Dictionary with script_path and script_content.
    """
    spec = IPIInputSpec(
        template_type=template_type,
        modifications=modifications or {},
        simulation_attrs=simulation_attrs or {},
    ).validate()

    xml_content = _select_template(spec.template_type)

    if spec.modifications:
        xml_content = _modify_xml_template(xml_content, spec.modifications)

    if spec.simulation_attrs:
        root = ET.fromstring(xml_content)
        for attr, value in spec.simulation_attrs.items():
            root.set(attr, str(value))
        xml_content = ET.tostring(root, encoding="unicode")

    os.makedirs("ipi_scripts", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    script_path = f"ipi_scripts/input_{timestamp}.xml"
    artifact = WorkflowArtifact(
        kind="ipi_input_xml",
        path=script_path,
        content=xml_content,
        metadata={"template_type": spec.template_type},
    ).validate()

    with open(artifact.path, "w", encoding="utf-8") as file:
        file.write(artifact.content)

    return {"script_path": artifact.path, "script_content": artifact.content}
