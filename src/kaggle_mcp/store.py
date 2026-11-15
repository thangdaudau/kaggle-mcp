"""Local registry of submitted jobs and uploaded datasets (one JSON file)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

TERMINAL = frozenset({"complete", "error", "cancel_acknowledged"})


@dataclass
class Job:
    id: str  # kernel slug
    kernel_ref: str  # owner/slug
    url: str
    project_dir: str
    entry: str
    created_at: str
    status: str = "queued"
    failure_message: str = ""
    datasets: list[str] = field(default_factory=lambda: list[str]())  # all attached dataset refs
    own_dataset: str | None = None  # per-job data dataset created by this tool
    ephemeral: bool = True
    output_dir: str | None = None
    cleaned: bool = False

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL


@dataclass
class Dataset:
    ref: str
    source_dir: str
    created_at: str


def default_home() -> Path:
    env = os.environ.get("KAGGLE_MCP_HOME")
    if env:
        return Path(env)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "kaggle-mcp"


class Store:
    def __init__(self, home: Path | None = None) -> None:
        self.path = (home or default_home()) / "jobs.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"jobs": [], "datasets": []}
        return cast(dict[str, Any], json.loads(self.path.read_text(encoding="utf-8")))

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def jobs(self) -> list[Job]:
        return [Job(**j) for j in self._load()["jobs"]]

    def datasets(self) -> list[Dataset]:
        return [Dataset(**d) for d in self._load()["datasets"]]

    def get_job(self, job_id: str) -> Job:
        for job in self.jobs():
            if job_id in (job.id, job.kernel_ref):
                return job
        raise KeyError(f"unknown job: {job_id}")

    def put_job(self, job: Job) -> None:
        data = self._load()
        jobs = [j for j in data["jobs"] if j["id"] != job.id]
        jobs.append(asdict(job))
        data["jobs"] = jobs
        self._save(data)

    def put_dataset(self, ds: Dataset) -> None:
        data = self._load()
        data["datasets"] = [d for d in data["datasets"] if d["ref"] != ds.ref] + [asdict(ds)]
        self._save(data)

    def remove_dataset(self, ref: str) -> None:
        data = self._load()
        data["datasets"] = [d for d in data["datasets"] if d["ref"] != ref]
        self._save(data)
