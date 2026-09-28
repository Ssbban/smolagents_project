import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from smolagents.tools import tool

from tools.specs import IPIASEClientSpec, WorkflowArtifact


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
    kwargs_str = json.dumps(dict(spec.calculator_kwargs), indent=4)
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
    calculator: str,
    unixsocket_name: str = "driver",
    init_xyz_path: str = "init.xyz",
    use_stress: bool = True,
    calculator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generate a validated ASE socket client script for an i-PI simulation.

    The socket name must match the i-PI input XML forcefield address.

    Args:
        calculator: ASE calculator name. Supported values are mace_mp and mace_off.
        unixsocket_name: UNIX socket name that must match the i-PI forcefield address.
        init_xyz_path: Path to the initial ASE-readable structure file.
        use_stress: Whether the ASE client should provide stress to i-PI.
        calculator_kwargs: Keyword arguments passed to the selected calculator factory.

    Returns:
        Dictionary with script_path and script_content.
    """
    spec = IPIASEClientSpec(
        calculator=calculator,
        unixsocket_name=unixsocket_name,
        init_xyz_path=init_xyz_path,
        use_stress=use_stress,
        calculator_kwargs=calculator_kwargs or {},
    ).validate(supported_calculators=CALCULATOR_IMPORTS)

    script_content = render_ase_client(spec)

    os.makedirs("ipi_scripts", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    script_path = f"ipi_scripts/run_ase_{timestamp}.py"
    artifact = WorkflowArtifact(
        kind="ipi_ase_client",
        path=script_path,
        content=script_content,
        metadata={"calculator": spec.calculator, "unixsocket_name": spec.unixsocket_name},
    ).validate()

    with open(artifact.path, "w", encoding="utf-8") as file:
        file.write(artifact.content)

    return {"script_path": artifact.path, "script_content": artifact.content}
