"""The MACE option table: extraction from arg_parser.py, where it comes from, and the look-up tool."""

import json
import sys
from pathlib import Path

import pytest

import tools.mace_cli_schema as schema_module
from tools.mace_cli_schema import (
    job_environment,
    load_mace_cli_schema,
    mace_cli_schema,
    parse_mace_arg_parser,
)

DATA = Path(__file__).parent / "data"


def source(version):
    return (DATA / f"mace_arg_parser_v{version}.py.txt").read_text(encoding="utf-8")


def table(version):
    return parse_mace_arg_parser(source(version), source_url="test", version=version)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")

    monkeypatch.setattr(schema_module.requests, "get", refuse)


@pytest.fixture(autouse=True)
def cache_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_module, "CACHE_DIR", tmp_path / ".cache" / "mace")


@pytest.mark.parametrize("version", ["0.3.13", "0.3.17"])
def test_only_the_training_parser_is_read(version):
    t = table(version)
    assert {"name", "train_file", "E0s", "r_max", "swa", "clip_grad"} <= set(t.flags)
    for preprocess_only in ["num_process", "shuffle", "compute_statistics", "config"]:
        assert t.get(preprocess_only) is None
    assert "unknown MACE option 'num_process'" in t.problem("num_process", 4)
    batch = t.get("batch_size")  # defined in both builders; the training one counts
    assert (batch.default, batch.help) == (10, "batch size")


def test_types_choices_aliases_and_nargs():
    t = table("0.3.17")
    assert t.get("stage_two") is t.get("swa") and t.get("swa").kind == "switch"
    assert t.get("--r-max").name == "r_max" and t.get("r_max").type_name == "float"
    assert t.get("plot").type_name == "bool" and t.get("clip_grad").type_name == "float_or_none"
    assert t.get("les_arguments").type_name == "yaml"
    assert t.get("device").choices == ["cpu", "cuda", "mps", "xpu"]
    assert t.get("foundation_model_readout").nargs == "?" and t.get("wandb_log_hypers").nargs == "+"
    assert t.get("foundation_model_readout").help.startswith("Transfer the foundation model's readout weights. Pass")


def test_two_versions_give_two_tables():
    old, new = table("0.3.13"), table("0.3.17")
    assert new.get("launcher") is not None and old.get("launcher") is None
    assert old.problem("launcher", "slurm").startswith("unknown MACE option 'launcher' (MACE 0.3.13)")
    assert new.problem("launcher", "slurm") is None
    old_names, new_names = (set(t.names("value") + t.names("switch")) for t in (old, new))
    assert new_names - old_names == {"launcher", "les_arguments"} and old_names <= new_names
    # 0.3.13 declared foundation_model_readout as store_false; 0.3.17 gives it an optional value
    assert old.get("foundation_model_readout").kind == "switch" and new.get("foundation_model_readout").kind == "value"


@pytest.mark.parametrize(
    "option, value, switch, expected",
    [
        ("r_max", "5.0", False, None),  # M04: argparse converts the text
        ("r_max", 5, False, None),
        ("batch_size", "10", False, None),
        ("batch_size", "ten", False, "invalid int value 'ten'"),
        ("batch_size", 10.0, False, "invalid int value '10.0'"),
        ("plot", "yes", False, None),
        ("plot", False, False, None),
        ("plot", "maybe", False, "expects a boolean"),
        ("clip_grad", "None", False, None),
        ("clip_grad", "none", False, "invalid float_or_none value"),
        ("device", "tpu", False, "must be one of ['cpu', 'cuda', 'mps', 'xpu']"),
        ("model", "MACE", False, None),
        ("swa", True, False, "is a switch without a value"),
        ("r_max", None, True, "needs a value"),
        ("stage_two", None, True, None),
        ("foundation_model_readout", None, True, None),
        ("learning_rat", 0.01, False, "unknown MACE option"),
    ],
)
def test_values_are_judged_as_argparse_would(option, value, switch, expected):
    problem = table("0.3.17").problem(option, value, switch=switch)
    assert problem is None if expected is None else expected in problem


def test_a_file_without_the_training_builder_is_refused():
    with pytest.raises(ValueError, match="no build_default_arg_parser"):
        parse_mace_arg_parser("def build_preprocess_arg_parser():\n    pass\n", source_url="x")
    with pytest.raises(ValueError, match="is not Python"):
        parse_mace_arg_parser("<!DOCTYPE html><html>", source_url="x")


# --- where the table comes from ------------------------------------------------------------------


def install_mace(site: Path, version: str, parser_version: str) -> None:
    """A fake MACE installation: arg_parser.py, a dist-info, and an __init__ that fails if imported."""
    package = site / "mace"
    (package / "tools").mkdir(parents=True)
    (package / "__init__.py").write_text("raise ImportError('MACE must not be imported')\n")
    (package / "tools" / "arg_parser.py").write_text(source(parser_version), encoding="utf-8")
    dist = site / f"mace_torch-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: mace-torch\nVersion: {version}\n")


@pytest.fixture
def conda(tmp_path):
    """A conda base with environments 'mace' (MACE 0.3.13 installed) and 'other' (no MACE); returns conda.sh."""
    base = tmp_path / "conda"
    (base / "etc" / "profile.d").mkdir(parents=True)
    (base / "etc" / "profile.d" / "conda.sh").write_text("")
    install_mace(base / "envs" / "mace" / "lib" / "python3.11" / "site-packages", "0.3.13", "0.3.13")
    (base / "envs" / "other" / "lib" / "python3.11" / "site-packages").mkdir(parents=True)
    return (base / "etc" / "profile.d" / "conda.sh").as_posix()


@pytest.fixture
def importable(tmp_path, monkeypatch):
    """MACE 0.3.17 importable where the agent runs."""
    site = tmp_path / "here"
    install_mace(site, "0.3.17", "0.3.17")
    monkeypatch.syspath_prepend(str(site))


@pytest.fixture
def nothing_importable(monkeypatch):
    monkeypatch.setattr(schema_module.importlib.util, "find_spec", lambda name: None)


def test_the_jobs_conda_environment_comes_first(conda, importable):
    t = load_mace_cli_schema(conda, "mace")
    assert (t.version, t.origin) == ("0.3.13", "installed in conda environment 'mace'")
    assert t.get("launcher") is None and "mace" not in sys.modules


def test_then_mace_importable_here_without_importing_it(conda, importable):
    t = load_mace_cli_schema(conda, "other")
    assert (t.version, t.origin) == ("0.3.17", "installed where the agent runs")
    assert t.get("launcher") is not None and "mace" not in sys.modules


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_then_the_pinned_release_tag_fetched_once_and_cached(session, nothing_importable, monkeypatch):
    urls = []

    def get(url, timeout):
        urls.append((url, timeout))
        return FakeResponse(source("0.3.17"))

    monkeypatch.setattr(schema_module.requests, "get", get)
    first = load_mace_cli_schema()
    monkeypatch.setattr(schema_module, "_TABLES", {})
    second = load_mace_cli_schema()
    assert urls == [("https://raw.githubusercontent.com/ACEsuit/mace/v0.3.17/mace/tools/arg_parser.py", 30)]
    assert first.version == second.version == "0.3.17" and first.origin.startswith("release tag v0.3.17")
    assert Path(".cache/mace/arg_parser-v0.3.17.py").read_text(encoding="utf-8") == source("0.3.17")


def test_an_unreachable_release_is_a_readable_error(session, nothing_importable, monkeypatch):
    def get(url, timeout):
        raise schema_module.requests.ConnectionError("offline")

    monkeypatch.setattr(schema_module.requests, "get", get)
    with pytest.raises(ValueError, match="could not be fetched from https://raw.githubusercontent.com/.*/v0.3.17/"):
        load_mace_cli_schema()
    assert not Path(".cache/mace").exists()


def test_missing_conda_facts_come_from_the_probe_run_once(session, conda, monkeypatch):
    probes = []

    def probe():
        probes.append(1)
        return {"available": True, "conda_activation_path": conda, "available_conda_envs": ["base", "mace", "other"]}

    monkeypatch.setattr(schema_module, "probe_cluster", probe)
    path, env, notes = job_environment(None, None)
    assert (path, env) == (conda, "mace") and len(notes) == 2 and "only environment with MACE" in notes[1]
    assert job_environment(None, "other")[:2] == (conda, "other")
    assert job_environment("/given/conda.sh", "given") == ("/given/conda.sh", "given", [])
    assert probes == [1]


def test_no_environment_is_guessed_when_two_have_mace(session, conda, monkeypatch):
    install_mace(Path(conda).parents[2] / "envs" / "other" / "lib" / "python3.11" / "site-packages", "0.3.17", "0.3.17")
    session.cluster = {"available": True, "conda_activation_path": conda, "available_conda_envs": ["mace", "other"]}
    assert job_environment(None, None)[1] is None


def test_the_look_up_asks_which_environment_when_two_have_mace(session, conda, nothing_importable):
    # Seen on the first cluster: three environments with MACE 0.3.10 to 0.3.13, and the 0.3.17 table was shown.
    install_mace(Path(conda).parents[2] / "envs" / "other" / "lib" / "python3.11" / "site-packages", "0.3.17", "0.3.17")
    session.cluster = {"available": True, "conda_activation_path": conda, "available_conda_envs": ["mace", "other"]}
    with pytest.raises(ValueError, match=r"several conda environments have MACE.*: mace \(0\.3\.13\), other \(0\.3\.17\)"):
        mace_cli_schema(names=["r_max"])
    assert mace_cli_schema(names=["r_max"], conda_env="mace")["mace_version"] == "0.3.13"


# --- the look-up tool ----------------------------------------------------------------------------


@pytest.fixture
def lookup(session, monkeypatch):
    session.cluster = {"available": False, "conda_activation_path": None, "available_conda_envs": []}
    monkeypatch.setattr(schema_module, "load_mace_cli_schema", lambda *a, **k: table("0.3.17"))


def test_summary_lists_names_only(lookup):
    result = mace_cli_schema()
    t = table("0.3.17")
    assert result["mace_version"] == "0.3.17" and result["source"] == "test"
    assert "swa" in result["switch_options"] and "r_max" in result["value_options"]
    assert len(result["value_options"]) + len(result["switch_options"]) == 22 == len({id(f) for f in t.flags.values()})
    assert len(json.dumps(result)) < 30 * 22 + 200  # names, no help texts (v1 sent the whole table every call)
    assert "help" not in json.dumps(result)


def test_names_describe_those_options(lookup):
    result = mace_cli_schema(names=["r_max", "stage_two", "learning_rat"])
    assert result["options"]["r_max"] == {"kind": "value", "type": "float", "default": 5.0,
                                          "help": "distance cutoff (in Ang)"}
    assert result["options"]["swa"]["aliases"] == ["stage_two"]
    assert list(result["unknown"]) == ["learning_rat"]
    assert "value_options" not in result and len(json.dumps(result)) < 1000


def test_search_finds_names(lookup):
    assert mace_cli_schema(search="FILE")["matches"] == [
        "les_arguments", "log_dir", "statistics_file", "test_file", "train_file", "valid_file"]
