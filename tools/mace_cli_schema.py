"""The options of `mace_run_train` for the MACE version that will run the job.

The table is read with `ast` from the body of `build_default_arg_parser` in MACE's `mace/tools/arg_parser.py`
(the module also defines the parser of `mace_prepare_data`, whose options `mace_run_train` rejects). MACE is
never imported: that would pull in torch, and the file may belong to another environment.

Where the file comes from, first that works:
  1. MACE installed in the job's conda environment, when that environment is visible from this machine;
  2. MACE importable where the agent runs;
  3. the release tag of DEFAULT_MACE_VERSION, fetched once from GitHub and kept under .cache/mace/.
"""

from __future__ import annotations

import ast
import copy
import difflib
import importlib.metadata
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
from smolagents.tools import tool

from tools.get_server_info import probe_cluster
from tools.session import get_session

DEFAULT_MACE_VERSION = "0.3.17"
TAG_URL = "https://raw.githubusercontent.com/ACEsuit/mace/v{version}/mace/tools/arg_parser.py"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "mace"
BUILDER = "build_default_arg_parser"
# MACE's argparse `type=` functions -> the type names shown to the model.
TYPE_NAMES = {"str": "str", "int": "int", "float": "float", "str2bool": "bool",
              "check_float_or_none": "float_or_none", "read_yaml": "yaml"}  # fmt: skip
SWITCH_ACTIONS = {"store_true", "store_false"}
_NOT_LITERAL = object()


def normalize_mace_option(option: str) -> str:
    return str(option).strip().removeprefix("--").replace("-", "_")


def cli_text(value: Any) -> str:
    """The text argparse receives for a value, as the job script writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def convert(type_name: str, text: str) -> Tuple[bool, Any]:
    """(True, value) when argparse would accept `text` for this type, else (False, reason).

    str2bool and check_float_or_none follow their definitions at the end of MACE's arg_parser.py.
    """
    try:
        if type_name == "int":
            return True, int(text)
        if type_name in ("float", "float_or_none"):
            return True, float(text)
    except ValueError:
        if type_name == "float_or_none" and text == "None":
            return True, None
        return False, f"invalid {type_name} value {text!r}"
    if type_name == "bool":
        if text.lower() in ("yes", "true", "t", "y", "1"):
            return True, True
        if text.lower() in ("no", "false", "f", "n", "0"):
            return True, False
        return False, f"expects a boolean (true/false), got {text!r}"
    return True, text


@dataclass(frozen=True)
class MACEFlag:
    name: str  # as written after "--" in MACE's source
    aliases: Tuple[str, ...]
    kind: str  # "value" | "switch"
    type_name: str
    choices: Optional[list]
    default: Any
    help: str
    nargs: Any = None

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": self.kind, "type": self.type_name, "default": self.default, "help": self.help}
        if self.choices is not None:
            info["choices"] = self.choices
        if self.aliases:
            info["aliases"] = list(self.aliases)
        if self.nargs is not None:
            info["nargs"] = self.nargs
        return info


@dataclass(frozen=True)
class MACECliSchema:
    version: str
    origin: str  # where the table was read from
    flags: Mapping[str, MACEFlag]  # by normalised name and alias

    def get(self, option: str) -> Optional[MACEFlag]:
        return self.flags.get(normalize_mace_option(option))

    def names(self, kind: str) -> List[str]:
        return sorted({flag.name for flag in self.flags.values() if flag.kind == kind}, key=str.lower)

    def problem(self, option: str, value: Any = None, *, switch: bool = False) -> Optional[str]:
        """Why mace_run_train would reject this option (with this value), or None."""
        flag = self.get(option)
        if flag is None:
            close = difflib.get_close_matches(normalize_mace_option(option), list(self.flags), n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else "; find it with mace_cli_schema(search=...)"
            return f"unknown MACE option {option!r} (MACE {self.version}){hint}"
        if switch:
            if flag.kind == "switch" or flag.nargs == "?":
                return None
            return f"--{flag.name} needs a value: put it in additional_args, not additional_flag_args"
        if flag.kind == "switch":
            return f"--{flag.name} is a switch without a value: put it in additional_flag_args, not additional_args"
        ok, converted = convert(flag.type_name, cli_text(value))
        if not ok:
            return f"--{flag.name}: {converted}"
        if flag.choices is not None and converted not in flag.choices:
            return f"--{flag.name} must be one of {flag.choices}; got {value!r}"
        return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NOT_LITERAL


def _text(node: ast.AST) -> str:
    """String literals of a help expression joined ("a" "b", "a" + "b", f-strings without their fields)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _text(node.left) + _text(node.right)
    if isinstance(node, ast.JoinedStr):
        return "".join(_text(part) for part in node.values)
    return ""


def _flag(call: ast.Call) -> Optional[MACEFlag]:
    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    options = [value for value in map(_literal, call.args) if isinstance(value, str) and value.startswith("--")]
    if not options or "is_config_file" in keywords:
        return None
    action = _literal(keywords["action"]) if "action" in keywords else None
    type_node = keywords.get("type")
    type_name = getattr(type_node, "id", None) or getattr(type_node, "attr", None) or "str"
    choices = _literal(keywords["choices"]) if "choices" in keywords else None
    default = _literal(keywords["default"]) if "default" in keywords else None
    nargs = _literal(keywords["nargs"]) if "nargs" in keywords else None
    return MACEFlag(
        name=options[0][2:],
        aliases=tuple(option[2:] for option in options[1:]),
        kind="switch" if action in SWITCH_ACTIONS else "value",
        type_name="switch" if action in SWITCH_ACTIONS else TYPE_NAMES.get(type_name, type_name),
        choices=list(choices) if isinstance(choices, (list, tuple)) else None,
        default=ast.unparse(keywords["default"]) if default is _NOT_LITERAL else default,
        help=" ".join(_text(keywords["help"]).split()) if "help" in keywords else "",
        nargs=None if nargs is _NOT_LITERAL else nargs,
    )


def parse_mace_arg_parser(source: str, *, source_url: str = "", version: str = "unknown") -> MACECliSchema:
    """The options added in `build_default_arg_parser` of an arg_parser.py text (`source_url` = its origin)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"MACE's arg_parser.py ({source_url or 'given text'}) is not Python: {error}") from None
    builder = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == BUILDER), None)
    if builder is None:
        raise ValueError(f"MACE's arg_parser.py ({source_url or 'given text'}) has no {BUILDER}()")
    flags: Dict[str, MACEFlag] = {}
    for node in ast.walk(builder):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in ("add_argument", "add"):
            flag = _flag(node)
            for name in (flag.name, *flag.aliases) if flag else ():
                flags[normalize_mace_option(name)] = flag
    return MACECliSchema(version=version, origin=source_url, flags=flags)


_TABLES: Dict[str, MACECliSchema] = {}


def _read_table(path: Path, version: str, origin: str) -> MACECliSchema:
    """Parse a local arg_parser.py once per file version (path and modification time)."""
    key = f"{path.resolve()}|{path.stat().st_mtime_ns}|{version}|{origin}"
    if key not in _TABLES:
        _TABLES[key] = parse_mace_arg_parser(path.read_text(encoding="utf-8"), source_url=origin, version=version)
    return _TABLES[key]


def _in_site_packages(env_dir: Path) -> Optional[Tuple[Path, str]]:
    """(arg_parser.py, version) of the MACE installed in a Python environment directory, or None."""
    for site in [*sorted(env_dir.glob("lib/python3*/site-packages")), env_dir / "Lib" / "site-packages"]:
        parser = site / "mace" / "tools" / "arg_parser.py"
        if parser.is_file():
            dist = sorted(site.glob("mace_torch-*.dist-info"))
            version = dist[-1].name.removeprefix("mace_torch-").removesuffix(".dist-info") if dist else "unknown"
            return parser, version
    return None


def conda_env_dir(conda_activation_path: Optional[str], conda_env: Optional[str]) -> Optional[Path]:
    """Directory of a conda environment, located from conda.sh (<base>/etc/profile.d/conda.sh) and the name;
    None when it is not visible from this machine."""
    if not conda_activation_path or not conda_env:
        return None
    parents = Path(os.path.expanduser(conda_activation_path)).parents
    if len(parents) < 3:
        return None
    base = parents[2]
    candidates = [base] if conda_env == "base" else [base / "envs" / conda_env, Path.home() / ".conda/envs" / conda_env]
    return next((env_dir for env_dir in candidates if env_dir.is_dir()), None)


def mace_in_conda_env(conda_activation_path: Optional[str], conda_env: Optional[str]) -> Optional[Tuple[Path, str]]:
    """(arg_parser.py, version) of the MACE installed in a conda environment, or None."""
    env_dir = conda_env_dir(conda_activation_path, conda_env)
    return _in_site_packages(env_dir) if env_dir else None


def mace_envs(conda_activation_path: Optional[str]) -> Dict[str, str]:
    """{environment: MACE version} of the conda environments of the cluster probe that have MACE installed."""
    envs = (get_session().cluster or {}).get("available_conda_envs") or []
    found = {env: mace_in_conda_env(conda_activation_path, env) for env in envs}
    return {env: hit[1] for env, hit in found.items() if hit}


def _importable_here() -> Optional[Tuple[Path, str]]:
    spec = importlib.util.find_spec("mace")  # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        return None
    parser = Path(list(spec.submodule_search_locations)[0]) / "tools" / "arg_parser.py"
    if not parser.is_file():
        return None
    try:
        version = importlib.metadata.version("mace-torch")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return parser, version


def _release_table(version: str) -> MACECliSchema:
    cached = CACHE_DIR / f"arg_parser-v{version}.py"
    origin = f"release tag v{version}; the job's environment was not visible from here"
    if not cached.is_file():
        url = TAG_URL.format(version=version)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise ValueError(f"MACE {version}'s option table could not be fetched from {url}: {error}") from None
        parse_mace_arg_parser(response.text, source_url=url, version=version)  # refuse to cache something else
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(response.text, encoding="utf-8")
    return _read_table(cached, version, origin)


def load_mace_cli_schema(
    conda_activation_path: Optional[str] = None, conda_env: Optional[str] = None
) -> MACECliSchema:
    """The option table of the MACE that will run the job (see the module docstring for the order)."""
    found = mace_in_conda_env(conda_activation_path, conda_env)
    if found:
        return _read_table(found[0], found[1], f"installed in conda environment '{conda_env}'")
    found = _importable_here()
    if found:
        return _read_table(found[0], found[1], "installed where the agent runs")
    return _release_table(DEFAULT_MACE_VERSION)


def job_environment(
    conda_activation_path: Optional[str], conda_env: Optional[str]
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Fill a missing conda.sh path or environment from the session's cluster probe (run once if needed).

    The environment is taken only when exactly one environment of the probe has MACE installed.
    Returns (conda_activation_path, conda_env, notes saying what was taken).
    """
    if conda_activation_path and conda_env:
        return conda_activation_path, conda_env, []
    session = get_session()
    if session.cluster is None:
        session.cluster = probe_cluster()
    notes = []
    if not conda_activation_path and session.cluster.get("conda_activation_path"):
        conda_activation_path = session.cluster["conda_activation_path"]
        notes.append(f"conda_activation_path {conda_activation_path} was taken from the cluster probe")
    if not conda_env and conda_activation_path:
        with_mace = list(mace_envs(conda_activation_path))
        if len(with_mace) == 1:
            conda_env = with_mace[0]
            notes.append(f"conda_env {conda_env} was taken from the cluster probe: the only environment with MACE")
    return conda_activation_path, conda_env, notes


def describe_mace_envs(conda_activation_path: Optional[str]) -> str:
    """'env (version), ...' of the environments with MACE, for messages."""
    return ", ".join(f"{env} ({version})" for env, version in mace_envs(conda_activation_path).items())


@tool
def mace_cli_schema(
    names: Optional[List[str]] = None, search: Optional[str] = None, conda_env: Optional[str] = None
) -> Dict[str, Any]:
    """
    Look up options of mace_run_train in the MACE version the job will run; MACE_script validates against the
    same table. Without arguments: {"mace_version", "source", "value_options", "switch_options"} with all option
    names (value options go into MACE_script's additional_args, switches into additional_flag_args). With names:
    {"options": {name: {kind, type, choices, default, help}}, "unknown": {name: close matches}}. With search:
    {"matches": [names whose name or help contains the text]}. Look up the options you need instead of guessing.

    Args:
        names: Option names to describe, e.g. ["r_max", "swa"].
        search: Text to find in option names and help texts, e.g. "stress".
        conda_env: The conda environment the job will use; its MACE version is then described. Needed when several
            environments have MACE (the tool says which).
    """
    conda_activation_path, conda_env, _ = job_environment(None, conda_env)
    if not conda_env and len(mace_envs(conda_activation_path)) > 1:
        raise ValueError(f"several conda environments have MACE, with different options: "
                         f"{describe_mace_envs(conda_activation_path)}. Call again with conda_env set to the one the "
                         "job will use (ask the person if they did not say).")
    table = load_mace_cli_schema(conda_activation_path, conda_env)
    result: Dict[str, Any] = {"mace_version": table.version, "source": table.origin}
    if names:
        found = {name: table.get(name) for name in names}
        result["options"] = {flag.name: flag.describe() for flag in found.values() if flag}
        result["unknown"] = {
            name: difflib.get_close_matches(normalize_mace_option(name), list(table.flags), n=3)
            for name, flag in found.items() if flag is None
        }  # fmt: skip
    if search:
        text = search.lower()
        result["matches"] = sorted(
            {f.name for f in table.flags.values() if text in f.name.lower() or text in f.help.lower()}, key=str.lower
        )
    if not names and not search:
        result["value_options"] = table.names("value")
        result["switch_options"] = table.names("switch")
    return copy.deepcopy(result)  # choices and defaults are lists of the cached table MACE_script validates with
