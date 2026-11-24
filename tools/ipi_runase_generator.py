import json
from typing import Any, Dict, Optional

from smolagents.tools import tool

from tools.session import get_session, unused_name
from tools.specs import IPIASEClientSpec, check, raise_problems


CALCULATOR_IMPORTS = {
    "mace_mp": "from mace.calculators import mace_mp",
    "mace_off": "from mace.calculators import mace_off",
}
CALCULATOR_FACTORIES = {
    "mace_mp": "mace_mp(**calculator_kwargs)",
    "mace_off": "mace_off(**calculator_kwargs)",
}


def render_ase_client(spec: IPIASEClientSpec) -> str:
    spec.validate(supported_calculators=CALCULATOR_IMPORTS)
    calc_name_lower = spec.calculator.lower()
    # Through JSON so that only plain values reach repr(): an object's own __repr__ could write any code here.
    kwargs_str = repr(json.loads(json.dumps(dict(spec.calculator_kwargs))))
    imports_line = CALCULATOR_IMPORTS[calc_name_lower]
    factory_line = CALCULATOR_FACTORIES[calc_name_lower]

    return f"""from ase.io import read
from ase.calculators.socketio import SocketClient
{imports_line}

INIT_XYZ = {json.dumps(spec.init_xyz_path)}
UNIX_SOCKET = {json.dumps(spec.unixsocket_name)}
USE_STRESS = {spec.use_stress}
CALCULATOR_KWARGS = {kwargs_str}


print(f"Reading atoms object from '{{INIT_XYZ}}'.")
atoms = read(INIT_XYZ, index=0)

print(f"Setting up calculator: '{calc_name_lower}' with arguments: {{CALCULATOR_KWARGS}}")
calculator_kwargs = CALCULATOR_KWARGS
atoms.calc = {factory_line}

print(f"Setting up socket client to connect to '{{UNIX_SOCKET}}'.")
client = SocketClient(unixsocket=UNIX_SOCKET)

print(f"Running client. Will provide forces and stress={{USE_STRESS}} to i-PI.")
client.run(atoms, use_stress=USE_STRESS)

print("i-PI simulation finished. ASE client is shutting down.")
"""


@tool
def ipi_runase_generator(
    ipi_input_path: str,
    calculator: str,
    unixsocket_name: str = "driver",
    init_xyz_path: str = "init.xyz",
    use_stress: bool = True,
    calculator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate a validated ASE socket client script for an i-PI simulation.

    The client is written as run_ase.py into the job bundle of the given i-PI input;
    the result is {"script_path", "bundle", "script_content"}.
    The socket name must match the i-PI input XML forcefield address.

    Args:
        ipi_input_path: script_path returned by ipi_InputFile_generator for this simulation.
        calculator: ASE calculator name. Supported values are mace_mp and mace_off.
        unixsocket_name: UNIX socket name that must match the i-PI forcefield address.
        init_xyz_path: Path of the initial ASE-readable structure on the cluster; a relative path is read from the job's bundle directory.
        use_stress: Whether the ASE client should provide stress to i-PI.
        calculator_kwargs: Keyword arguments passed to the selected calculator factory (plain values only).

    Returns:
        Dictionary with script_path, bundle and script_content.
    """
    session = get_session()
    spec = IPIASEClientSpec(
        calculator=calculator,
        unixsocket_name=unixsocket_name,
        init_xyz_path=init_xyz_path,
        use_stress=use_stress,
        calculator_kwargs=calculator_kwargs or {},
    )
    problems = spec.problems(supported_calculators=CALCULATOR_IMPORTS)
    ipi_input = check(problems, session.require_artifact, ipi_input_path, "ipi_input")
    raise_problems("ipi_runase_generator", problems)

    script_content = render_ase_client(spec)
    name = unused_name(ipi_input.bundle, "run_ase.py")
    artifact = session.write_artifact(ipi_input.bundle, name, script_content, "ase_client")
    return {"script_path": artifact.path, "bundle": artifact.bundle, "script_content": script_content}
