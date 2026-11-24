from pathlib import Path, PurePosixPath

import pytest

from tools.ipi_InputFile_generator import ipi_InputFile_generator
from tools.ipi_runase_generator import ipi_runase_generator
from tools.ipi_submit_generator import ipi_submit_generator
from tools.preflight import preflight

TEMPLATE = """<simulation verbosity="medium">
  <ffsocket name="driver" mode="unix" pbc="False"><address> driver </address></ffsocket>
  <total_steps> 100 </total_steps>
  <system><initialize nbeads="1"><file mode="ase"> init.xyz </file></initialize></system>
</simulation>"""

SUBMIT = dict(job_name="ipi_water", time="02:00:00", partition="cpu", gpus_per_node=0, memory="8G",
              conda_env="ipi", conda_path="/opt/conda/etc/profile.d/conda.sh")


@pytest.fixture
def template(monkeypatch):
    monkeypatch.setattr("tools.ipi_InputFile_generator._select_template", lambda template_type: TEMPLATE)


def test_input_bundle_client_and_submit_script_belong_together(session, template):
    xml = ipi_InputFile_generator("02-NVT-Langevin", {"steps": {"xpath": "./total_steps", "text": "500"}})
    assert xml["script_path"] == f"{xml['bundle']}/input.xml"
    client = ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp")
    assert client["bundle"] == xml["bundle"] and client["script_path"].endswith("/run_ase.py")
    job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    assert job["bundle"] == xml["bundle"] and job["script_path"].endswith("/job.sh")
    text = job["script_content"]
    assert "i-pi input.xml &> log.i-pi &" in text and "python run_ase.py" in text
    assert "ipi_scripts" not in text and "runs/" not in text
    assert b"\r\n" not in Path(job["script_path"]).read_bytes()
    assert session.find_artifact(job["script_path"]).kind == "ipi_submit"


def test_preflight_checks_the_files_of_the_bundle(session, template):
    xml = ipi_InputFile_generator("00-NVE")
    ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp")
    job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    report = preflight(script_path=job["script_path"])
    assert report["ok"] and report["errors"] == [] and len(report["warnings"]) == 1
    Path(xml["script_path"]).write_text("<simulation/>")
    report = preflight(script_path=job["script_path"])
    assert not report["ok"] and report["errors"] == [f"{xml['script_path']} was changed or removed after it was generated"]


def test_preflight_needs_the_client_named_in_the_script(session, template):
    xml = ipi_InputFile_generator("00-NVE")
    bundle = xml["bundle"]
    script = session.write_artifact(bundle, "job.sh", "#!/bin/bash\n#SBATCH --job-name=a\n#SBATCH --time=10\n"
                                    "i-pi input.xml &\n", "ipi_submit")
    errors = preflight(script_path=script.path)["errors"]
    assert errors == [f"the script does not run a ase_client file generated in its bundle {bundle}"]


def test_second_client_and_second_job_get_new_names(session, template):
    xml = ipi_InputFile_generator("00-NVE")
    ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_mp")
    second = ipi_runase_generator(ipi_input_path=xml["script_path"], calculator="mace_off")
    assert PurePosixPath(second["script_path"]).name == "run_ase-2.py"
    first_job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    second_job = ipi_submit_generator(ipi_input_path=xml["script_path"], **SUBMIT)
    assert "python run_ase-2.py" in first_job["script_content"]
    assert PurePosixPath(second_job["script_path"]).name == "job-2.sh"


def test_two_inputs_in_the_same_second_give_two_bundles(session, template):
    first, second = ipi_InputFile_generator("00-NVE"), ipi_InputFile_generator("00-NVE")
    assert first["bundle"] != second["bundle"]


def test_client_needs_a_generated_input(session, template):
    with pytest.raises(ValueError, match="not a ipi_input file generated in this session"):
        ipi_runase_generator(ipi_input_path="input.xml", calculator="mace_mp")


def test_submit_reports_missing_client_and_spec_problems_together(session, template):
    xml = ipi_InputFile_generator("00-NVE")
    with pytest.raises(ValueError) as caught:
        ipi_submit_generator(ipi_input_path=xml["script_path"], **{**SUBMIT, "memory": "lots"})
    message = str(caught.value)
    assert "has no ASE client yet" in message and "memory must be" in message


def test_submit_refuses_client_of_another_bundle(session, template):
    first, second = ipi_InputFile_generator("00-NVE"), ipi_InputFile_generator("00-NVE")
    client = ipi_runase_generator(ipi_input_path=second["script_path"], calculator="mace_mp")
    with pytest.raises(ValueError, match="is not in the bundle of"):
        ipi_submit_generator(ipi_input_path=first["script_path"], ase_path=client["script_path"], **SUBMIT)
