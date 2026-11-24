import pytest

from config import DEFAULT_API_BASE, Settings, SettingsError

BASE = {"SMOLAGENTS_MODEL_ID": "some-model", "SMOLAGENTS_API_KEY": "sk-test"}


def settings(env, tmp_path):
    return Settings.from_env(env, dotenv_path=tmp_path / "absent.env")


def test_defaults(tmp_path):
    s = settings(BASE, tmp_path)
    assert (s.model_id, s.api_base, s.temperature, s.max_steps, s.planning_interval, s.runs_dir) == (
        "some-model", DEFAULT_API_BASE, None, 20, 10, "runs")
    assert "sk-test" not in repr(s) and "api_key" not in s.public()


def test_values_from_environment(tmp_path):
    env = {**BASE, "SMOLAGENTS_API_BASE": "http://localhost:8000/v1", "SMOLAGENTS_TEMPERATURE": "0.2",
           "SMOLAGENTS_MAX_STEPS": "8", "SMOLAGENTS_PLANNING_INTERVAL": "4", "SMOLAGENTS_RUNS_DIR": "out"}
    s = settings(env, tmp_path)
    assert (s.api_base, s.temperature, s.max_steps, s.planning_interval, s.runs_dir) == (
        "http://localhost:8000/v1", 0.2, 8, 4, "out")


@pytest.mark.parametrize("missing", ["SMOLAGENTS_MODEL_ID", "SMOLAGENTS_API_KEY"])
def test_missing_required_value_names_the_variable(tmp_path, missing):
    env = {**BASE, missing: "  "}
    with pytest.raises(SettingsError, match=f"^{missing} is not set"):
        settings(env, tmp_path)


@pytest.mark.parametrize("name, value", [("SMOLAGENTS_MAX_STEPS", "ten"), ("SMOLAGENTS_MAX_STEPS", "0"),
                                         ("SMOLAGENTS_TEMPERATURE", "warm")])
def test_bad_numbers(tmp_path, name, value):
    with pytest.raises(SettingsError, match=name):
        settings({**BASE, name: value}, tmp_path)


def test_dotenv_file_is_read_and_environment_wins(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("SMOLAGENTS_MODEL_ID=from-file\nSMOLAGENTS_API_KEY=file-key\nSMOLAGENTS_MAX_STEPS=5\n")
    s = Settings.from_env({"SMOLAGENTS_MAX_STEPS": "7"}, dotenv_path=dotenv)
    assert (s.model_id, s.api_key, s.max_steps) == ("from-file", "file-key", 7)
