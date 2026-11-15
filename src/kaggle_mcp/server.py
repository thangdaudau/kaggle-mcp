"""MCP server exposing Kaggle background jobs as tools."""

from __future__ import annotations

from dataclasses import asdict
from functools import cache
from typing import Any

from mcp.server import MCPServer

from kaggle_mcp.jobs import Jobs
from kaggle_mcp.kaggle_client import RealKaggleClient
from kaggle_mcp.store import Job, Store

INSTRUCTIONS = """\
Runs local Python projects as background jobs on Kaggle's free machines.
Jobs run on Kaggle, independent of this session: submit_job returns immediately.
To be notified when a job finishes, run the shell command `kjob wait <job_id>` in the
background (it polls every 60s and exits when the job is done), then call job_output.
Job state is kept locally, so after a session restart list_jobs shows earlier jobs.
"""

server = MCPServer("kaggle-jobs", instructions=INSTRUCTIONS)


@cache
def _jobs() -> Jobs:
    return Jobs(RealKaggleClient(), Store())


def _job_view(job: Job) -> dict[str, Any]:
    view = asdict(job)
    view["finished"] = job.terminal
    return view


@server.tool()
def submit_job(
    project_dir: str,
    entry: str,
    data_dir: str | None = None,
    datasets: list[str] | None = None,
    gpu: bool = False,
    internet: bool = True,
    ephemeral: bool = True,
) -> dict[str, Any]:
    """Run a Python project on Kaggle as a private background job; returns immediately.

    The project's code (excluding .git, virtualenvs, caches and data_dir) is bundled; on Kaggle
    `entry` runs as `__main__` with cwd=/kaggle/working. Files written there become the job's
    output (limit ~20 GB). If the project has requirements.txt it is pip-installed first
    (needs internet=True). Each run is limited to 12 hours.

    Args:
        project_dir: Local project directory.
        entry: Script to run, relative to project_dir (e.g. "train.py").
        data_dir: Optional local directory uploaded as a private dataset for this job only.
            On Kaggle its path is in env var KJOB_DATA_DIR. Subdirectories are uploaded zipped.
        datasets: Existing Kaggle dataset refs ("owner/slug") to attach, mounted under
            /kaggle/input. Use upload_dataset for data shared by many jobs.
        gpu: Request a GPU (counts against the weekly GPU quota).
        internet: Allow network access (pip installs, downloads).
        ephemeral: Delete the job's kernel, outputs and per-job dataset from Kaggle after
            job_output downloads them, so Kaggle storage does not fill up.
    """
    job = _jobs().submit(
        project_dir,
        entry,
        data_dir=data_dir,
        datasets=datasets,
        gpu=gpu,
        internet=internet,
        ephemeral=ephemeral,
    )
    return _job_view(job) | {"hint": f"run `kjob wait {job.id}` in the background to be notified"}


@server.tool()
def job_status(job_id: str) -> dict[str, Any]:
    """Current status of a job: queued, running, complete, error or cancel_*."""
    return _job_view(_jobs().refresh(job_id))


@server.tool()
def list_jobs(refresh: bool = True) -> list[dict[str, Any]]:
    """All jobs submitted from this machine, newest first. refresh=True asks Kaggle for status."""
    return [_job_view(j) for j in _jobs().list(refresh=refresh)]


@server.tool()
def job_logs(job_id: str, tail_chars: int = 8000) -> str:
    """stdout/stderr of a job's latest run (last tail_chars characters; 0 = all).

    Kaggle usually publishes the log only after the run finishes.
    """
    return _jobs().logs(job_id, tail=tail_chars)


@server.tool()
def job_output(job_id: str, dest_dir: str | None = None) -> dict[str, Any]:
    """Download a finished job's output files (default: <project_dir>/kaggle_out/<job_id>/).

    For ephemeral jobs the kernel and its per-job dataset are then deleted from Kaggle.
    """
    job, files = _jobs().output(job_id, dest_dir)
    return {"job": _job_view(job), "files": files}


@server.tool()
def cleanup_job(job_id: str) -> dict[str, Any]:
    """Delete a finished job's kernel (with its outputs) and per-job dataset from Kaggle."""
    return _job_view(_jobs().cleanup(job_id))


@server.tool()
def upload_dataset(data_dir: str, name: str) -> dict[str, Any]:
    """Upload a local directory as a private Kaggle dataset to reuse across jobs.

    Pass the returned ref in submit_job(datasets=[...]). Not deleted automatically;
    use delete_dataset when done.
    """
    return asdict(_jobs().upload_dataset(data_dir, name))


@server.tool()
def list_datasets() -> list[dict[str, Any]]:
    """Datasets uploaded with upload_dataset that still exist."""
    return [asdict(d) for d in _jobs().store.datasets()]


@server.tool()
def delete_dataset(ref: str) -> str:
    """Delete a dataset created by upload_dataset. Refuses if an unfinished job uses it."""
    _jobs().delete_dataset(ref)
    return f"deleted {ref}"


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
