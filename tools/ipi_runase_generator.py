import json
from pathlib import Path
from typing import Any, Dict, Optional

from smolagents.tools import tool

from tools.ipi_check import links
from tools.session import get_session, unused_name
from tools.specs import IPIASEClientSpec, check, raise_problems


CALCULATOR_IMPORTS = {
    "mace_mp": "from mace.calculators import mace_mp",
    "mace_off": "from mace.calculators import mace_off",
}
CALCULATOR_FACTORIES = {
    "mace_mp": "mace_mp(device=DEVICE, **CALCULATOR_KWARGS)",
    "mace_off": "mace_off(device=DEVICE, **CALCULATOR_KWARGS)",
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
DEVICE = {json.dumps(spec.device)}
USE_STRESS = {spec.use_stress}
CALCULATOR_KWARGS = {kwargs_str}


print(f"Reading atoms object from '{{INIT_XYZ}}'.")
atoms = read(INIT_XYZ, index=0)

print(f"Setting up calculator: '{calc_name_lower}' on {{DEVICE}} with arguments: {{CALCULATOR_KWARGS}}")
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
    device: str = "cpu",
    use_stress: bool = True,
    calculator_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write the ASE socket client run_ase.py into the bundle of an i-PI input.

    The socket address and the structure file are read from that input.xml, so the client always
    connects to its own i-PI and starts from the same structure. The Slurm script requests a GPU
    only for a cuda client. Result: {"script_path", "bundle", "script_content"}.

    Args:
        ipi_input_path: script_path returned by ipi_InputFile_generator for this simulation.
        calculator: ASE calculator name: mace_mp or mace_off.
        device: Where the calculator runs: "cpu" or "cuda".
        use_stress: Whether the client returns the stress; needed by every constant-pressure (NPT) template.
        calculator_kwargs: Other keyword arguments of the calculator factory, e.g. {"model": "medium"} (plain values only).
    """
    session = get_session()
    problems: list[str] = []
    ipi_input = check(problems, session.require_artifact, ipi_input_path, "ipi_input")
    wiring = links(Path(ipi_input.path).read_text(encoding="utf-8")) if ipi_input else {}
    spec = IPIASEClientSpec(
        calculator=calculator,
        unixsocket_name=wiring.get("socket") or "driver",
        init_xyz_path=wiring.get("structure") or "init.xyz",
        use_stress=use_stress,
        calculator_kwargs=calculator_kwargs or {},
        device=device,
    )
    raise_problems("ipi_runase_generator", problems + spec.problems(supported_calculators=CALCULATOR_IMPORTS))

    script_content = render_ase_client(spec)
    name = unused_name(ipi_input.bundle, "run_ase.py")
    artifact = session.write_artifact(ipi_input.bundle, name, script_content, "ase_client")
    return {"script_path": artifact.path, "bundle": artifact.bundle, "script_content": script_content}
