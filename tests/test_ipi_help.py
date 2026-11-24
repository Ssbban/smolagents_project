from tools.ipi_help import ipi_tag_help


def test_thermostat_lists_its_modes_and_children():
    first = ipi_tag_help("thermostat")["matches"][0]
    assert first["paths"][0] == "simulation/system/motion/dynamics/thermostat"
    assert {"langevin", "svr", "pile_g", "gle"} <= set(first["attributes"]["mode"])
    assert "tau" in first["children"]


def test_values_with_a_dimension_say_so():
    timestep = ipi_tag_help("timestep")["matches"][0]
    assert timestep["dimension"] == "time" and "units" in timestep["attributes"]
    [temperature] = ipi_tag_help("temperature")["matches"]
    assert temperature["paths"] == ["simulation/system/ensemble/temperature"]
    assert temperature["dimension"] == "temperature" and temperature["type"] == "float"


def test_tau_of_molecular_dynamics_comes_before_rarer_motions():
    first_paths = [match["paths"][0] for match in ipi_tag_help("tau")["matches"][:2]]
    assert sorted(first_paths) == ["simulation/system/motion/dynamics/barostat/tau",
                                   "simulation/system/motion/dynamics/thermostat/tau"]


def test_attribute_by_path():
    result = ipi_tag_help("system/initialize@nbeads")
    [match] = result["matches"]
    assert match["kind"] == "attribute" and match["type"] == "int" and "number of beads" in match["help"]
    assert result["ipi_version"] == "3.3.1"


def test_changing_an_answer_does_not_change_the_next_one():
    ipi_tag_help("thermostat")["matches"][0]["attributes"]["mode"].append("bogus")
    assert "bogus" not in ipi_tag_help("thermostat")["matches"][0]["attributes"]["mode"]


def test_unknown_tag():
    result = ipi_tag_help("bogus")
    assert result["matches"] == [] and "no such tag" in result["note"]
