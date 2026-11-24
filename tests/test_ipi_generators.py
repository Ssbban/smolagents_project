import re
from pathlib import Path, PurePosixPath

import pytest

from tools.ipi_InputFile_generator import ipi_InputFile_generator, ipi_template_schema
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.preflight import preflight

WATER = "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n"
SUBMIT = dict(job_name="ipi_water", time="02:00:00", partition="cpu", memory="8G",
              conda_env="ipi", conda_path="/opt/conda/etc/profile.d/conda.sh")  # fmt: skip
CLUSTER_UNKNOWN = "cluster facts unknown"
TEMPLATES = Path(__file__).resolve().parents[1] / "tools" / "ipi_templates"
LANGEVIN = "02-NVT-Langevin"
# Arguments of the nine input cases of the v1 baseline (U4 section 3.3).
BASELINE = {
    "I00_valid_temperature": {"modifications": {"set_T": {"xpath": "./system/ensemble/temperature", "text": "350"}}},
    "I01_xpath_no_match": {"modifications": {"x": {"xpath": ".//does_not_exist", "text": "1"}}},
    "I02_xpath_syntax_error": {"modifications": {"x": {"xpath": ".//[@", "text": "1"}}},
    "I03_malformed_value_steps": {"modifications": {"x": {"xpath": "./total_steps", "text": "-5 abc"}}},
    "I04_unrestricted_ffsocket_to_inet": {"modifications": {
        "x": {"xpath": "./ffsocket", "attrs": {"mode": "inet"}},
        "y": {"xpath": "./ffsocket/address", "text": "attacker.example.com"}}},
    "I05_bad_sim_attr": {"simulation_attrs": {"verbosity": "loud"}},
    "I06_unknown_template": {"template_type": "99-NOPE"},
    "I07_append_block": {"modifications": {"add": {"parent_xpath": ".", "tag": "prng",
                                                   "fields": [{"tag": "seed", "text": "12345"}]}}},
    "I08_text_with_xml_markup": {"modifications": {
        "x": {"xpath": "./system/ensemble/temperature", "text": "</temperature><evil/>"}}},
}  # fmt: skip


@pytest.fixture
def water(session):
    Path("water.xyz").write_text(WATER)
    return "water.xyz"


def make_job(template="02-NVT-Langevin", structure="water.xyz", client=None, **submit):
    xml = ipi_InputFile_generator(template, structure)
    ase = ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp", **(client or {}))
    job = ipi_submit_generator(ipi_input_path=xml["script_path"], **{**SUBMIT, **submit})
    return xml, ase, job


def line(text, key):
    return re.search(rf"^{key}.*$", text, re.M)[0]


# The nine input cases of the v1 baseline (U4 section 3.3), with the outcome they have now.
CASES = {
    "I00_valid_temperature": None,
    "I01_xpath_no_match": "'.//does_not_exist' is not an editable path",
    "I02_xpath_syntax_error": "'.//[@' is not an editable path",
    "I03_malformed_value_steps": "invalid literal for int() with base 10: '-5 abc'",
    "I04_unrestricted_ffsocket_to_inet": "'./ffsocket' is not an editable path",
    "I05_bad_sim_attr": "simulation_attrs.verbosity must be one of",
    "I06_unknown_template": "Unknown template type: 99-NOPE",
    "I07_append_block": "<prng> appears 2 times in <simulation>; i-PI reads only the last one",
    "I08_text_with_xml_markup": "i-PI: ValueError",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_baseline_input_cases(water, case):
    args = {"template_type": LANGEVIN, **BASELINE[case]}
    if CASES[case] is None:
        result = ipi_InputFile_generator(**args, structure_path=water)
        assert result["facts"]["temperature_K"] == pytest.approx([350])
        assert "<temperature units=\"kelvin\">350</temperature>" in result["script_content"]
    else:
        with pytest.raises(ValueError) as caught:
            ipi_InputFile_generator(**args, structure_path=water)
        assert CASES[case] in str(caught.value)


def test_redirecting_the_socket_lists_both_refused_paths(water):
    args = BASELINE["I04_unrestricted_ffsocket_to_inet"]
    with pytest.raises(ValueError, match="2 problem") as caught:
        ipi_InputFile_generator(LANGEVIN, water, **args)
    assert "'./ffsocket/address' is not an editable path" in str(caught.value)


def test_schema_lists_the_catalogue_and_only_editable_paths():
    catalogue = ipi_template_schema()["templates"]
    assert sorted(entry["name"] for entry in catalogue) == sorted(p.name for p in TEMPLATES.iterdir() if p.is_dir())
    assert all(entry["what"] and entry["when"] for entry in catalogue)
    schema = ipi_template_schema("07-REMD-NVT")
    xpaths = [entry["xpath"] for entry in schema["paths"]]
    assert "./system_template/instance[3]" in xpaths and "./total_steps" in xpaths
    assert not any("ffsocket" in x or x.endswith(("/initialize/file", "/forces/force")) for x in xpaths)
    assert "." not in xpaths and "55d6ef7" in schema["source"]


def test_an_appended_output_is_accepted_and_a_forcefield_is_not(water):
    block = {"parent_xpath": "./output", "tag": "trajectory", "attrs": {"filename": "for", "stride": "200"},
             "text": "forces"}
    result = ipi_InputFile_generator("02-NVT-Langevin", water, {"forces": block})
    assert "<trajectory filename=\"for\" stride=\"200\">forces</trajectory>" in result["script_content"]
    socket = {"parent_xpath": ".", "tag": "ffsocket", "attrs": {"mode": "inet"}}
    with pytest.raises(ValueError, match="cannot be added: the force field is the template's socket"):
        ipi_InputFile_generator("02-NVT-Langevin", water, {"socket": socket})


def test_sockets_prefix_is_refused(water):
    with pytest.raises(ValueError, match="sockets_prefix is not accepted"):
        ipi_InputFile_generator("00-NVE", water, simulation_attrs={"sockets_prefix": "/scratch/"})


def test_socket_and_structure_agree_across_the_three_files(session, water):
    xml, ase, job = make_job()
    bundle = xml["bundle"]
    address = f"{session.id}-{PurePosixPath(bundle).name}"
    assert xml["facts"]["forcefields"][0]["address"] == address and xml["facts"]["structure"] == "water.xyz"
    assert f"<address>{address}</address>" in xml["script_content"]
    assert line(ase["script_content"], "UNIX_SOCKET") == f'UNIX_SOCKET = "{address}"'
    assert line(job["script_content"], "SOCKET=") == f"SOCKET=/tmp/ipi_{address}"
    assert "<file mode=\"ase\">water.xyz</file>" in xml["script_content"]
    assert line(ase["script_content"], "INIT_XYZ") == 'INIT_XYZ = "water.xyz"'
    staged = session.find_artifact(f"{bundle}/water.xyz")
    assert staged.kind == "structure" and Path(staged.path).read_text() == WATER
    assert {a.bundle for a in session.artifacts} == {bundle}


def test_two_inputs_get_two_sockets(water):
    first, second = ipi_InputFile_generator("00-NVE", water), ipi_InputFile_generator("00-NVE", water)
    assert first["bundle"] != second["bundle"]
    assert first["facts"]["forcefields"][0]["address"] != second["facts"]["forcefields"][0]["address"]


def test_a_structure_not_readable_here_must_be_absolute(session, water):
    with pytest.raises(ValueError, match="is not a file on this machine"):
        ipi_InputFile_generator("00-NVE", "data/missing.xyz")
    with pytest.raises(ValueError, match="does not expand '~'"):
        ipi_InputFile_generator("00-NVE", "~/missing-structure-file.xyz")
    xml, ase, job = make_job(structure="/cluster/data/water.xyz")
    assert "<file mode=\"ase\">/cluster/data/water.xyz</file>" in xml["script_content"]
    assert line(ase["script_content"], "INIT_XYZ") == 'INIT_XYZ = "/cluster/data/water.xyz"'
    assert not any(a.kind == "structure" for a in session.artifacts)
    report = preflight(script_path=job["script_path"])
    assert report["ok"] and any("/cluster/data/water.xyz is not readable here" in w for w in report["warnings"])


def test_only_structure_files_are_staged(session, water):
    Path("notes.txt").write_text(WATER)
    for path in ["notes.txt", "~/.ssh/id_rsa"]:
        with pytest.raises(ValueError, match="is not a structure file ASE recognises"):
            ipi_InputFile_generator("00-NVE", path)
    assert session.artifacts == []


def test_job_script_fails_on_errors_waits_for_the_socket_and_needs_no_source_checkout(water):
    _, _, job = make_job()
    text = job["script_content"]
    lines = text.splitlines()
    last_sbatch = max(i for i, item in enumerate(lines) if item.startswith("#SBATCH"))
    assert lines[last_sbatch + 2] == "set -euo pipefail"
    assert "sleep 30" not in text and "env.sh" not in text and '[ -S "$SOCKET" ] && break' in text
    assert "i-pi input.xml &> log.i-pi &" in text and "    python run_ase.py &" in text
    assert 'wait "$IPI_PID"' in text and "CLIENTS=1" in text
    assert "runs/" not in text and b"\r\n" not in Path(job["script_path"]).read_bytes()


def test_optional_environment_script(water):
    _, _, job = make_job(ipi_env_path="~/i-pi/env.sh")
    assert "source ~/i-pi/env.sh\nsource /opt/conda/etc/profile.d/conda.sh" in job["script_content"]
    # set -u is off while scripts not written for it run (env.sh, conda.sh, conda activate)
    lines = job["script_content"].splitlines()
    plus, minus = lines.index(next(x for x in lines if x.startswith("set +u"))), lines.index("set -u")
    assert all(plus < i < minus for i, x in enumerate(lines) if x.startswith(("source ", "conda activate")))


def test_gpu_request_follows_the_client_device(water):
    _, _, cpu_job = make_job()
    assert "--gres" not in cpu_job["script_content"] and cpu_job["gpus_per_node"] == 0
    _, ase, gpu_job = make_job(client={"device": "cuda"})
    assert line(ase["script_content"], "DEVICE") == 'DEVICE = "cuda"'
    assert "#SBATCH --gres=gpu:1" in gpu_job["script_content"]
    with pytest.raises(ValueError, match="runs on cuda but gpus_per_node is 0"):
        ipi_submit_generator(ipi_input_path=ase["script_path"].replace("run_ase.py", "input.xml"),
                             gpus_per_node=0, **SUBMIT)  # fmt: skip


def test_device_belongs_in_its_own_argument(water):
    xml = ipi_InputFile_generator("00-NVE", water)
    with pytest.raises(ValueError, match="set the device with the device argument"):
        ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp",
                             calculator_kwargs={"device": "cuda"})  # fmt: skip


@pytest.mark.parametrize(
    "template, clients, fragment",
    [
        ("02-NVT-Langevin", 2, "num_force_providers must be 1: the input has one bead and one replica"),
        ("07-REMD-NVT", 6, "num_force_providers must be at most 5"),
        ("09-PINVT-for-dynamics", 33, "num_force_providers must be at most 32"),
    ],
)
def test_client_count_is_bounded_by_beads_times_replicas(water, template, clients, fragment):
    xml = ipi_InputFile_generator(template, water)
    ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp")
    with pytest.raises(ValueError, match=fragment):
        ipi_submit_generator(ipi_input_path=xml["script_path"], num_force_providers=clients, **SUBMIT)
    job = ipi_submit_generator(ipi_input_path=xml["script_path"], num_force_providers=clients - 1, **SUBMIT)
    assert f"CLIENTS={clients - 1}" in job["script_content"]


def test_preflight_passes_a_generated_job(water):
    _, _, job = make_job("09-PINVT-for-dynamics", num_force_providers=4)
    report = preflight(script_path=job["script_path"])
    assert report["ok"] and report["errors"] == [] and len(report["warnings"]) == 1
    assert CLUSTER_UNKNOWN in report["warnings"][0]


def _forged_job(session, job, script_edit=None, input_edit=None):
    """A new bundle whose files are ledger artefacts but were not made by the generators as they are."""
    xml_text = Path(job["script_path"]).with_name("input.xml").read_text()
    bundle = session.new_bundle("forged")
    if input_edit:
        xml_text = input_edit(xml_text)
    session.write_artifact(bundle, "water.xyz", WATER, "structure")
    session.write_artifact(bundle, "input.xml", xml_text, "ipi_input")
    client = Path(job["script_path"]).with_name("run_ase.py").read_text()
    session.write_artifact(bundle, "run_ase.py", client, "ase_client")
    script = job["script_content"]
    return session.write_artifact(bundle, "job.sh", script_edit(script) if script_edit else script, "ipi_submit")


def test_preflight_finds_a_socket_mismatch(session, water):
    _, _, job = make_job()
    forged = _forged_job(session, job, input_edit=lambda t: re.sub(r"<address>[^<]*", "<address>other", t))
    errors = preflight(script_path=forged.path)["errors"]
    assert errors == [f"{forged.bundle}/run_ase.py connects to another socket than {forged.bundle}/input.xml"]


def test_preflight_finds_a_script_waiting_for_another_socket(session, water):
    _, _, job = make_job()
    forged = _forged_job(session, job, script_edit=lambda s: re.sub(r"^SOCKET=.*$", "SOCKET=/tmp/ipi_x", s, flags=re.M))
    assert preflight(script_path=forged.path)["errors"] == [
        f"{forged.path} waits for another socket than {forged.bundle}/input.xml opens"]


def test_preflight_finds_too_many_clients_and_an_input_that_does_not_load(session, water):
    _, _, job = make_job()
    forged = _forged_job(session, job, script_edit=lambda s: s.replace("CLIENTS=1", "CLIENTS=3"),
                         input_edit=lambda t: t.replace("<total_steps>20000", "<total_steps>many"))
    errors = preflight(script_path=forged.path)["errors"]
    assert len(errors) == 1 and "does not load in i-PI" in errors[0] and "'many'" in errors[0]
    forged = _forged_job(session, job, script_edit=lambda s: s.replace("CLIENTS=1", "CLIENTS=3"))
    assert preflight(script_path=forged.path)["errors"] == ["3 clients for 1 beads x replicas"]


def test_preflight_needs_the_structure_in_the_bundle(session, water):
    _, _, job = make_job()
    forged = _forged_job(session, job, input_edit=lambda t: t.replace("water.xyz", "ice.xyz"))
    errors = preflight(script_path=forged.path)["errors"]
    assert f"the structure 'ice.xyz' is not a staged file of {forged.bundle}" in errors
    assert f"{forged.bundle}/run_ase.py reads another structure than {forged.bundle}/input.xml" in errors


def test_preflight_checks_the_files_of_the_bundle(water):
    xml, _, job = make_job()
    Path(xml["script_path"]).write_text("<simulation/>")
    report = preflight(script_path=job["script_path"])
    assert f"{xml['script_path']} was changed or removed after it was generated" in report["errors"]


def test_second_client_and_second_job_get_new_names(water):
    xml = ipi_InputFile_generator("00-NVE", water)
    ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp")
    second = ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_off")
    assert PurePosixPath(second["script_path"]).name == "run_ase-2.py"
    first_job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    second_job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    assert "python run_ase-2.py &" in first_job["script_content"]
    assert PurePosixPath(second_job["script_path"]).name == "job-2.sh"
    assert preflight(script_path=second_job["script_path"])["ok"]


def test_client_needs_a_generated_input(session):
    with pytest.raises(ValueError, match="not a ipi_input file generated in this session"):
        ipi_runase_generator(ipi_input_path="input.xml", calculator="mace_mp")


def test_submit_reports_missing_client_and_spec_problems_together(water):
    xml = ipi_InputFile_generator("00-NVE", water)
    with pytest.raises(ValueError) as caught:
        ipi_submit_generator(ipi_input_path=xml["script_path"], **{**SUBMIT, "memory": "lots"})
    message = str(caught.value)
    assert "has no ASE client yet" in message and "memory must be" in message


def test_submit_refuses_client_of_another_bundle(water):
    first, second = ipi_InputFile_generator("00-NVE", water), ipi_InputFile_generator("00-NVE", water)
    client = ipi_runase_generator(ipi_input_path=second["script_path"], calculator="mace_mp")
    with pytest.raises(ValueError, match="is not in the bundle of"):
        ipi_submit_generator(ipi_input_path=first["script_path"], ase_path=client["script_path"], **SUBMIT)
