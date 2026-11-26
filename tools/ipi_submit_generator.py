import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

from smolagents.tools import tool

from tools.ipi_check import SOCKETS_DIR, check_input, links
from tools.mace_cli_schema import conda_env_dir, mace_in_conda_env
from tools.preflight import Problem, register
from tools.session import Artifact, Session, get_session, unused_name
from tools.specs import ClusterSpec, IPISubmitSpec, check, raise_problems

SOCKET_WAIT_SECONDS = 300
CLIENT_LINE_RE = {key: re.compile(rf'^{key} = "(.*)"$', re.M) for key in ("UNIX_SOCKET", "INIT_XYZ", "DEVICE")}
SCRIPT_LINE_RE = {
    "socket": re.compile(rf"^SOCKET={re.escape(SOCKETS_DIR)}(\S+)$", re.M),
    "clients": re.compile(r"^CLIENTS=(\d+)$", re.M),
    "input": re.compile(r"^i-pi (\S+)", re.M),
    "client": re.compile(r"^\s*python (\S+) &$", re.M),
}


def render_ipi_submit_script(spec: IPISubmitSpec) -> str:
    spec.validate()
    cluster = spec.cluster
    header = cluster.sbatch_header(cluster.gpus_per_node)
    environment = cluster.environment(*filter(None, [spec.ipi_env_path]))
    body = [
        f"SOCKET={SOCKETS_DIR}{spec.socket_address}",
        f"CLIENTS={spec.num_force_providers}",
        "# Unbuffered, so that log.i-pi and the client's output show the run while it goes (else only at the end).",
        "export PYTHONUNBUFFERED=1",
        "",
        "# A socket file left by an earlier run of this job (requeue, crash) would stop i-PI from starting.",
        'rm -f "$SOCKET"',
        f"i-pi {spec.ipi_input_path} &> log.i-pi &",
        "IPI_PID=$!",
        f"for second in $(seq {SOCKET_WAIT_SECONDS}); do",
        '    [ -S "$SOCKET" ] && break',
        '    kill -0 "$IPI_PID" 2>/dev/null || break',
        "    sleep 1",
        "done",
        'if [ ! -S "$SOCKET" ]; then',
        '    echo "i-PI did not open $SOCKET; see log.i-pi" >&2',
        '    kill "$IPI_PID" 2>/dev/null || true',
        "    exit 1",
        "fi",
        "",
        "CLIENT_PIDS=()",
        'for client in $(seq "$CLIENTS"); do',
        f"    python {spec.ase_path} &",
        '    CLIENT_PIDS+=("$!")',
        "done",
        "# A failing client or i-PI ends the job with its error code (set -e).",
        'for pid in "${CLIENT_PIDS[@]}"; do',
        '    wait "$pid"',
        "done",
        'wait "$IPI_PID"',
    ]
    return "\n".join(header + ["", "set -euo pipefail", ""] + environment + [""] + body) + "\n"


def _bundle_files(session: Session, ipi_input_path: str, ase_path: Optional[str]) -> tuple[Artifact, Artifact]:
    """The i-PI input and the ASE client of one bundle; ValueError if they do not belong together."""
    ipi_input = session.require_artifact(ipi_input_path, "ipi_input")
    if ase_path:
        client = session.require_artifact(ase_path, "ase_client")
        if client.bundle != ipi_input.bundle:
            raise ValueError(
                f"{client.path} is not in the bundle of {ipi_input.path}; generate the client with "
                f"ipi_runase_generator(ipi_input_path={ipi_input.path!r}, ...)"
            )
        return ipi_input, client
    clients = [a for a in session.artifacts if a.kind == "ase_client" and a.bundle == ipi_input.bundle]
    if not clients:
        raise ValueError(
            f"the bundle of {ipi_input.path} has no ASE client yet; call "
            f"ipi_runase_generator(ipi_input_path={ipi_input.path!r}, ...) first"
        )
    return ipi_input, clients[-1]


def _client_value(client: Artifact, key: str) -> Optional[str]:
    match = CLIENT_LINE_RE[key].search(Path(client.path).read_text(encoding="utf-8"))
    return match[1] if match else None


def _max_clients(facts: Dict[str, Any]) -> int:
    return facts.get("nbeads", 1) * facts.get("replicas", 1)


def _has_ipi(conda_path: Optional[str], env: str) -> bool:
    env_dir = conda_env_dir(conda_path, env)
    return env_dir is not None and (env_dir / "bin" / "i-pi").is_file()


def _environment_problems(session: Session, conda_path: Optional[str], conda_env: Optional[str],
                          ipi_env_path: Optional[str]) -> list[str]:
    """What the job's conda environment lacks, when it is visible from here: i-pi (unless ipi_env_path provides
    it), and MACE for run_ase.py."""
    if conda_env_dir(conda_path, conda_env) is None:
        return []
    problems = []
    if not ipi_env_path and not _has_ipi(conda_path, conda_env):
        with_ipi = [env for env in (session.cluster or {}).get("available_conda_envs", []) if _has_ipi(conda_path, env)]
        problems.append(f"conda environment {conda_env!r} has no i-pi; environments with i-pi: {with_ipi}")
    if not mace_in_conda_env(conda_path, conda_env):
        problems.append(f"conda environment {conda_env!r} has no MACE, which the ASE client imports")
    return problems


@tool
def ipi_submit_generator(
    ipi_input_path: str,
    ase_path: Optional[str] = None,
    job_name: str = "I-PI",
    time: str = "24:00:00",
    partition: str = None,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
    memory: str = "20G",
    conda_env: str = None,
    conda_path: str = None,
    ipi_env_path: Optional[str] = None,
    num_force_providers: int = 1,
    additional_slurm_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write the Slurm script job.sh that runs i-PI and its ASE client(s) in the bundle of an i-PI input.

    The script waits for i-PI's socket before it starts the clients and fails when i-PI or a client
    fails. Result: {"script_path", "bundle", "script_content", "num_force_providers", "gpus_per_node"};
    submit it with slurm_submit(script_path).

    Args:
        ipi_input_path: script_path returned by ipi_InputFile_generator.
        ase_path: script_path returned by ipi_runase_generator for that input; default: the latest client in its bundle.
        job_name: Slurm job name.
        time: Slurm time limit, e.g. "24:00:00" or "2-00:00:00" (any sbatch form).
        partition: Slurm partition name.
        nodes: Number of Slurm nodes.
        ntasks: Number of Slurm tasks.
        cpus_per_task: CPU cores shared by i-PI and the client(s), e.g. 4 to 8 for a cpu client. Left out: one core.
        gpus_per_node: GPUs to request; default: 1 for a cuda client, 0 for a cpu client.
        memory: Slurm memory allocation, for example 20G.
        conda_env: Conda environment with i-PI, ASE and the calculator (mace-torch) installed; checked when visible from here.
        conda_path: Path to conda.sh.
        ipi_env_path: Optional script to source before conda (e.g. an i-PI source checkout's env.sh); usually not needed.
        num_force_providers: Number of client processes. One is enough for a single classical system; up to
            beads x replicas of the input (path integrals, replica exchange) can work in parallel.
        additional_slurm_args: Additional Slurm options as long option name -> single value, e.g. {"account": "proj1"}.
    """
    session = get_session()
    problems: list[str] = []
    ipi_input, client = check(problems, _bundle_files, session, ipi_input_path, ase_path) or (None, None)
    facts: Dict[str, Any] = {}
    socket, device = "driver", "cpu"
    if ipi_input:
        xml = Path(ipi_input.path).read_text(encoding="utf-8")
        result = check_input(xml)
        problems += [f"{ipi_input.path}: {problem}" for problem in result["problems"]]
        facts, socket = result["facts"], links(xml)["socket"] or socket
    if client:
        device = _client_value(client, "DEVICE") or device
    if gpus_per_node is None:
        gpus_per_node = 1 if device == "cuda" else 0
    cluster = ClusterSpec(
        job_name=job_name,
        time=time,
        partition=partition,
        nodes=nodes,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        gpus_per_node=gpus_per_node,
        memory=memory,
        conda_env=conda_env,
        conda_activation_path=conda_path,
        additional_slurm_args=additional_slurm_args or {},
    )
    spec = IPISubmitSpec(
        cluster=cluster,
        ipi_input_path=PurePosixPath(ipi_input.path).name if ipi_input else "input.xml",
        ase_path=PurePosixPath(client.path).name if client else "run_ase.py",
        socket_address=socket,
        ipi_env_path=ipi_env_path or None,
        num_force_providers=num_force_providers,
        max_force_providers=_max_clients(facts),
        device=device,
    )
    problems += _environment_problems(session, conda_path, conda_env, ipi_env_path)
    raise_problems("ipi_submit_generator", problems + spec.problems())

    script_content = render_ipi_submit_script(spec)
    name = unused_name(ipi_input.bundle, "job.sh")
    artifact = session.write_artifact(ipi_input.bundle, name, script_content, "ipi_submit")
    return {
        "script_path": artifact.path,
        "bundle": artifact.bundle,
        "script_content": script_content,
        "num_force_providers": spec.num_force_providers,
        "gpus_per_node": gpus_per_node,
    }


def check_ipi_job(artifact: Artifact, session: Session) -> list[Problem]:
    """Pre-flight for an i-PI job: the input still loads, socket and structure agree, the client count fits."""
    script = Path(artifact.path).read_text(encoding="utf-8")
    members = {PurePosixPath(a.path).name: a for a in session.artifacts if a.bundle == artifact.bundle}
    found = {key: pattern.search(script) for key, pattern in SCRIPT_LINE_RE.items()}
    ipi_input = members.get(found["input"][1]) if found["input"] else None
    client = members.get(found["client"][1]) if found["client"] else None
    if not ipi_input or not client or ipi_input.kind != "ipi_input" or client.kind != "ase_client":
        return []  # check_bundle reports a script that does not run a generated input and client
    xml = Path(ipi_input.path).read_text(encoding="utf-8")
    result, wiring = check_input(xml), links(xml)
    problems = [Problem("error", f"{ipi_input.path} does not load in i-PI: {p}") for p in result["problems"]]
    if wiring["socket"] is None:
        problems.append(Problem("error", f"{ipi_input.path} has no single unix socket"))
    elif _client_value(client, "UNIX_SOCKET") != wiring["socket"]:
        problems.append(Problem("error", f"{client.path} connects to another socket than {ipi_input.path}"))
    elif not found["socket"] or found["socket"][1] != wiring["socket"]:
        problems.append(Problem("error", f"{artifact.path} waits for another socket than {ipi_input.path} opens"))
    if _client_value(client, "INIT_XYZ") != wiring["structure"]:
        problems.append(Problem("error", f"{client.path} reads another structure than {ipi_input.path}"))
    clients = int(found["clients"][1]) if found["clients"] else 1
    if "replicas" in result["facts"] and clients > _max_clients(result["facts"]):
        problems.append(Problem("error", f"{clients} clients for {_max_clients(result['facts'])} beads x replicas"))
    structure = wiring["structure"] or ""
    if "/" not in structure:
        staged = members.get(structure)
        if staged is None or staged.kind != "structure":
            problems.append(Problem("error", f"the structure {structure!r} is not a staged file of {artifact.bundle}"))
    elif not Path(structure).is_file() and session.cluster and session.cluster.get("available"):
        problems.append(Problem("error", f"the structure {structure} does not exist"))
    elif not Path(structure).is_file():
        problems.append(Problem("warning", f"the structure {structure} is not readable here; it must exist on the "
                                           "compute nodes"))
    return problems


register("ipi_submit", check_ipi_job)
