"""Session ledger: the files the generator tools wrote, the jobs submitted from them, the cluster probe.

The current session is a module global on purpose: generated code can read attributes of tool objects,
but not the globals of this module.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.trace import Trace, utc_now


@dataclass(frozen=True)
class Artifact:
    kind: str  # "mace_script" | "ipi_input" | "structure" | "ase_client" | "ipi_submit"
    path: str  # POSIX, relative to the working directory
    sha256: str
    bundle: str  # directory of the job this file belongs to (POSIX, relative)
    created_at: str


@dataclass
class Job:
    job_id: str
    script: str
    bundle: str
    submitted_at: str
    state: str = "SUBMITTED"


def relative_posix(path: str | Path) -> str:
    """`path` relative to the working directory with forward slashes (absolute if it lies outside)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def unused_name(bundle: str | Path, filename: str) -> str:
    """`filename`, or `stem-2.ext`, `stem-3.ext`, ... if the bundle already has it."""
    stem, dot, suffix = filename.rpartition(".")
    if not dot:
        stem, suffix = filename, ""
    candidate, number = filename, 1
    while (Path(bundle) / candidate).exists():
        number += 1
        candidate = f"{stem}-{number}{dot}{suffix}"
    return candidate


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Session:
    def __init__(self, runs_dir: str | Path = "runs"):
        self.id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        self.dir = Path(runs_dir) / self.id
        self.dir.mkdir(parents=True)
        self.trace = Trace(self.dir / "trace.jsonl", self.id)
        self.cluster: dict | None = None
        self._ledger: list[Artifact | Job] = []
        self._bundles = 0

    @property
    def artifacts(self) -> list[Artifact]:
        return [item for item in self._ledger if isinstance(item, Artifact)]

    @property
    def jobs(self) -> list[Job]:
        return [item for item in self._ledger if isinstance(item, Job)]

    def new_bundle(self, job_name: str) -> Path:
        self._bundles += 1
        bundle = self.dir / f"{self._bundles:02d}-{job_name}"
        bundle.mkdir()
        return bundle

    def write_artifact(self, bundle: str | Path, filename: str, content: str, kind: str) -> Artifact:
        path = Path(bundle) / filename
        data = content.replace("\r\n", "\n").encode("utf-8")
        try:
            with open(path, "xb") as file:
                file.write(data)
        except FileExistsError:
            raise FileExistsError(f"{relative_posix(path)} already exists; generated files are never overwritten") from None
        artifact = Artifact(kind, relative_posix(path), _sha256(data), relative_posix(bundle), utc_now())
        self._ledger.append(artifact)
        self.trace.emit("artifact_written", **asdict(artifact))
        return artifact

    def find_artifact(self, path: str) -> Artifact | None:
        key = relative_posix(str(path).strip().replace("\\", "/"))
        return next((a for a in reversed(self.artifacts) if a.path == key), None)

    def require_artifact(self, path: str, kind: str) -> Artifact:
        """The artefact at `path` of the given kind, or a ValueError the model can act on."""
        artifact = self.find_artifact(path)
        if artifact is None or artifact.kind != kind:
            raise ValueError(
                f"{path!r} is not a {kind} file generated in this session; "
                "pass the script_path returned by the tool that generated it"
            )
        return artifact

    def is_unchanged(self, artifact: Artifact) -> bool:
        path = Path(artifact.path)
        return path.is_file() and _sha256(path.read_bytes()) == artifact.sha256

    def record_job(self, artifact: Artifact, job_id: str) -> Job:
        job = Job(job_id, artifact.path, artifact.bundle, utc_now())
        self._ledger.append(job)
        return job

    def job_for(self, artifact: Artifact) -> Job | None:
        return next((j for j in self.jobs if j.script == artifact.path), None)

    def find_job(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.job_id == str(job_id).strip()), None)

    def mark(self) -> int:
        return len(self._ledger)

    def since(self, mark: int) -> dict:
        items = self._ledger[mark:]
        return {
            "artifacts": [asdict(i) for i in items if isinstance(i, Artifact)],
            "jobs": [asdict(i) for i in items if isinstance(i, Job)],
        }

    def snapshot(self) -> dict:
        bundles: dict[str, list[dict]] = {}
        for artifact in self.artifacts:
            bundles.setdefault(artifact.bundle, []).append({"path": artifact.path, "kind": artifact.kind})
        cluster = None
        if self.cluster is not None:
            cluster = {
                "available": self.cluster.get("available"),
                "partitions": [p["name"] for p in self.cluster.get("partitions", [])],
            }
        return {
            "session": self.id,
            "bundles": bundles,
            "jobs": [{"job_id": j.job_id, "script": j.script, "state": j.state} for j in self.jobs],
            "cluster": cluster,
        }


_current: Session | None = None


def start_session(runs_dir: str | Path = "runs") -> Session:
    global _current
    _current = Session(runs_dir)
    return _current


def get_session() -> Session:
    if _current is None:
        raise RuntimeError("no session")
    return _current
