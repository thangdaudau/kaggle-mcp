"""Job operations, independent of the MCP layer so they can be tested with a fake client."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from kaggle_mcp import bundle
from kaggle_mcp.kaggle_client import KaggleClient
from kaggle_mcp.store import Dataset, Job, Store

DATASET_READY_TIMEOUT_S = 30 * 60
LOG_TAIL_CHARS = 8000


class Jobs:
    def __init__(
        self,
        client: KaggleClient,
        store: Store,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.client = client
        self.store = store
        self._sleep = sleep
        self._now = now

    # ---- datasets -------------------------------------------------------

    def _new_slug(self, prefix: str, name: str) -> str:
        stamp = self._now().strftime("%Y%m%d-%H%M%S")
        base = bundle.slugify(name)[: 50 - len(prefix) - len(stamp) - 2] or "job"
        return f"{prefix}-{base}-{stamp}".replace("--", "-")

    def _upload_dir(self, data_dir: Path, slug: str) -> str:
        """Upload data_dir as a private dataset and wait until Kaggle has processed it."""
        if not data_dir.is_dir():
            raise ValueError(f"data_dir is not a directory: {data_dir}")
        ref = f"{self.client.username()}/{slug}"
        with tempfile.TemporaryDirectory(prefix="kjob-ds-") as tmp:
            staging = Path(tmp)
            for entry in data_dir.iterdir():
                (staging / entry.name).symlink_to(entry.resolve())
            meta = {"id": ref, "title": slug, "licenses": [{"name": "CC0-1.0"}]}
            (staging / "dataset-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            error = self.client.create_dataset(str(staging))
        if error:
            raise RuntimeError(f"dataset upload failed: {error}")
        self._wait_dataset_ready(ref)
        return ref

    def _wait_dataset_ready(self, ref: str) -> None:
        deadline = time.monotonic() + DATASET_READY_TIMEOUT_S
        while True:
            status = self.client.dataset_status(ref)
            if status == "ready":
                return
            if status == "error":
                raise RuntimeError(f"Kaggle failed to process dataset {ref}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"dataset {ref} not ready after {DATASET_READY_TIMEOUT_S}s")
            self._sleep(5)

    def upload_dataset(self, data_dir: str, name: str) -> Dataset:
        path = Path(data_dir).expanduser().resolve()
        ref = self._upload_dir(path, self._new_slug("kjobd", name))
        ds = Dataset(ref=ref, source_dir=str(path), created_at=self._now().isoformat())
        self.store.put_dataset(ds)
        return ds

    def delete_dataset(self, ref: str) -> None:
        if ref not in {d.ref for d in self.store.datasets()}:
            raise ValueError(f"{ref} was not uploaded by kaggle-mcp; refusing to delete it")
        users = [j.id for j in self.store.jobs() if ref in j.datasets and not j.terminal]
        if users:
            raise ValueError(f"{ref} is used by unfinished jobs: {users}")
        self.client.delete_dataset(ref)
        self.store.remove_dataset(ref)

    # ---- jobs -----------------------------------------------------------

    def submit(
        self,
        project_dir: str,
        entry: str,
        *,
        data_dir: str | None = None,
        datasets: list[str] | None = None,
        gpu: bool = False,
        internet: bool = True,
        ephemeral: bool = True,
    ) -> Job:
        project = Path(project_dir).expanduser().resolve()
        if not (project / entry).is_file():
            raise ValueError(f"entry script not found: {project / entry}")
        data_path = Path(data_dir).expanduser().resolve() if data_dir else None
        if data_path and not data_path.is_dir():
            raise ValueError(f"data_dir is not a directory: {data_path}")
        code_zip = bundle.build_code_zip(project, exclude=data_path)

        slug = self._new_slug("kjob", project.name)
        own_dataset = (
            self._upload_dir(data_path, slug.replace("kjob-", "kjobd-", 1)) if data_path else None
        )
        sources = [*(datasets or []), *([own_dataset] if own_dataset else [])]
        data_slug = own_dataset.split("/", 1)[1] if own_dataset else None
        kernel_ref = f"{self.client.username()}/{slug}"

        with tempfile.TemporaryDirectory(prefix="kjob-k-") as tmp:
            bundle.write_kernel_folder(
                Path(tmp),
                kernel_ref=kernel_ref,
                script=bundle.render_bootstrap(code_zip, entry, data_slug),
                gpu=gpu,
                internet=internet,
                dataset_sources=sources,
            )
            result = self.client.push_kernel(tmp)
        if result.error:
            if own_dataset:
                self.client.delete_dataset(own_dataset)
            raise RuntimeError(f"kernel push failed: {result.error}")

        job = Job(
            id=slug,
            kernel_ref=kernel_ref,
            url=result.url,
            project_dir=str(project),
            entry=entry,
            created_at=self._now().isoformat(),
            datasets=sources,
            own_dataset=own_dataset,
            ephemeral=ephemeral,
        )
        self.store.put_job(job)
        return job

    def refresh(self, job_id: str) -> Job:
        job = self.store.get_job(job_id)
        if job.cleaned or job.terminal:
            return job
        st = self.client.kernel_status(job.kernel_ref)
        job.status, job.failure_message = st.status, st.failure_message
        self.store.put_job(job)
        return job

    def list(self, refresh: bool = True) -> list[Job]:
        jobs = self.store.jobs()
        if refresh:
            jobs = [self.refresh(j.id) for j in jobs]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def logs(self, job_id: str, tail: int = LOG_TAIL_CHARS) -> str:
        job = self.store.get_job(job_id)
        if job.cleaned:
            log_file = Path(job.output_dir or "") / f"{job.id}.log"
            if job.output_dir and log_file.exists():
                text = log_file.read_text(encoding="utf-8")
            else:
                raise ValueError(f"job {job.id} was cleaned up and no local log exists")
        else:
            text = self.client.kernel_log(job.kernel_ref)
        return text[-tail:] if tail > 0 else text

    def output(self, job_id: str, dest_dir: str | None = None) -> tuple[Job, list[str]]:
        job = self.refresh(job_id)
        if not job.terminal:
            raise ValueError(f"job {job.id} is still {job.status}; wait until it finishes")
        if job.cleaned:
            raise ValueError(
                f"job {job.id} was already downloaded to {job.output_dir} and cleaned up"
            )
        dest = (
            Path(dest_dir).expanduser()
            if dest_dir
            else Path(job.project_dir) / "kaggle_out" / job.id
        )
        dest.mkdir(parents=True, exist_ok=True)
        files = self.client.download_output(job.kernel_ref, str(dest))
        job.output_dir = str(dest.resolve())
        self.store.put_job(job)
        if job.ephemeral:
            self.cleanup(job.id)
            job = self.store.get_job(job.id)
        return job, [os.path.relpath(f, dest) for f in files]

    def cleanup(self, job_id: str) -> Job:
        """Delete the job's kernel (and its outputs) and its per-job dataset on Kaggle."""
        job = self.refresh(job_id)
        if not job.terminal:
            raise ValueError(f"job {job.id} is still {job.status}; refusing to delete it")
        if job.cleaned:
            return job
        self.client.delete_kernel(job.kernel_ref)
        if job.own_dataset:
            self.client.delete_dataset(job.own_dataset)
        job.cleaned = True
        self.store.put_job(job)
        return job

    def wait(
        self, job_id: str, poll_s: float = 60, on_change: Callable[[Job], None] | None = None
    ) -> Job:
        last = ""
        while True:
            job = self.refresh(job_id)
            if job.status != last and on_change:
                on_change(job)
            last = job.status
            if job.terminal:
                return job
            self._sleep(poll_s)
