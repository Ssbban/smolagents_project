from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import requests
from smolagents.tools import tool


MACE_ARG_PARSER_URL = (
    "https://raw.githubusercontent.com/ACEsuit/mace/refs/heads/main/mace/tools/arg_parser.py"
)

VALUE_ACTIONS = {None, "store", "append", "extend"}
SWITCH_ACTIONS = {"store_true", "store_false"}


def normalize_mace_option(option: str) -> str:
    return option.strip().removeprefix("--").replace("-", "_")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _keyword_literal(call: ast.Call, name: str) -> Any:
    node = _keyword(call, name)
    return _literal(node) if node is not None else None


def _keyword_name(call: ast.Call, name: str) -> Optional[str]:
    node = _keyword(call, name)
    return _name(node) if node is not None else None


@dataclass(frozen=True)
class MACEFlag:
    canonical: str
    aliases: List[str]
    kind: str
    action: Optional[str]
    type_name: Optional[str]
    choices: Optional[List[Any]]
    required: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "aliases": self.aliases,
            "kind": self.kind,
            "action": self.action,
            "type": self.type_name,
            "choices": self.choices,
            "required": self.required,
        }


@dataclass(frozen=True)
class MACECliSchema:
    source_url: str
    flags: Mapping[str, MACEFlag]

    @property
    def value_options(self) -> List[str]:
        return sorted(name for name, flag in self.flags.items() if flag.kind == "value")

    @property
    def switch_options(self) -> List[str]:
        return sorted(name for name, flag in self.flags.items() if flag.kind == "switch")

    def resolve(self, option: str) -> MACEFlag:
        normalized = normalize_mace_option(option)
        if normalized in self.flags:
            return self.flags[normalized]
        raise ValueError(
            f"Unknown MACE CLI option '{option}'. Sync the schema from {self.source_url} and use an official flag."
        )

    def require_value(self, option: str) -> MACEFlag:
        flag = self.resolve(option)
        if flag.kind != "value":
            raise ValueError(
                f"MACE option '--{flag.canonical}' is a switch flag. Put it in additional_flag_args, not additional_args."
            )
        return flag

    def require_switch(self, option: str) -> MACEFlag:
        flag = self.resolve(option)
        if flag.kind != "switch":
            raise ValueError(
                f"MACE option '--{flag.canonical}' requires a value. Put it in additional_args, not additional_flag_args."
            )
        return flag

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "value_options": self.value_options,
            "switch_options": self.switch_options,
            "flags": {name: flag.as_dict() for name, flag in sorted(self.flags.items())},
        }


def parse_mace_arg_parser(source: str, *, source_url: str = MACE_ARG_PARSER_URL) -> MACECliSchema:
    tree = ast.parse(source)
    flags: Dict[str, MACEFlag] = {}
    alias_lookup: Dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"add_argument", "add"}:
            continue

        option_strings = [
            value
            for value in (_literal(arg) for arg in node.args)
            if isinstance(value, str) and value.startswith("--")
        ]
        if not option_strings:
            continue

        canonical = normalize_mace_option(option_strings[0])
        aliases = [normalize_mace_option(option) for option in option_strings]
        action = _keyword_literal(node, "action")
        if action in SWITCH_ACTIONS:
            kind = "switch"
        elif action in VALUE_ACTIONS:
            kind = "value"
        else:
            raise ValueError(
                f"Unsupported argparse action for MACE option '--{canonical}': {action}"
            )

        choices_literal = _keyword_literal(node, "choices")
        choices = list(choices_literal) if isinstance(choices_literal, (list, tuple)) else None
        flag = MACEFlag(
            canonical=canonical,
            aliases=aliases,
            kind=kind,
            action=action,
            type_name=_keyword_name(node, "type"),
            choices=choices,
            required=bool(_keyword_literal(node, "required")),
        )
        flags[canonical] = flag
        for alias in aliases:
            alias_lookup[alias] = canonical

    resolved_flags = dict(flags)
    for alias, canonical in alias_lookup.items():
        resolved_flags[alias] = flags[canonical]

    return MACECliSchema(source_url=source_url, flags=resolved_flags)


def load_mace_cli_schema(url: str = MACE_ARG_PARSER_URL) -> MACECliSchema:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_mace_arg_parser(response.text, source_url=url)


def validate_mace_value(flag: MACEFlag, value: Any) -> None:
    if flag.type_name == "str2bool":
        if isinstance(value, bool):
            return
        if isinstance(value, str) and value.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return
        raise ValueError(f"MACE option '--{flag.canonical}' requires an explicit boolean value.")
    if flag.type_name == "int" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"MACE option '--{flag.canonical}' requires an integer value.")
    if flag.type_name == "float" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"MACE option '--{flag.canonical}' requires a numeric value.")
    if flag.choices is not None and value not in flag.choices:
        raise ValueError(
            f"MACE option '--{flag.canonical}' must be one of {flag.choices}, got {value!r}."
        )


@tool
def mace_cli_schema() -> Dict[str, Any]:
    """
    Fetch and parse the official MACE training CLI schema.

    Args:
        None: This tool takes no arguments.

    Returns:
        Dictionary containing value_options, switch_options and per-flag metadata.
    """
    return load_mace_cli_schema().as_dict()
