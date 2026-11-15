from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from kaggle_mcp import bundle, cli
from kaggle_mcp.jobs import Jobs
from kaggle_mcp.kaggle_client import KernelStatus, PushResult
from kaggle_mcp.store import Job, Store


class FakeClient:
    def __init__(self) -> None:
        self.pushed: list[dict[str, object]] = []
        self.scripts: list[str] = []
        self.datasets: dict[str, list[str]] = {}  # ref -> uploaded file names
        self.kernels: set[str] = set()
        self.status = "running"
        self.push_error: str | None = None
        self.dataset_states = ["pending", "ready"]

    def username(self) -> str:
        return "me"

    def push_kernel(self, folder: str) -> PushResult:
        meta = json.loads((Path(folder) / "kernel-metadata.json").read_text())
        self.pushed.append(meta)
        self.scripts.append((Path(folder) / meta["code_file"]).read_text())
        if self.push_error:
            return PushResult(url="", version=None, error=self.push_error)
        self.kernels.add(meta["id"])
        return PushResult(url=f"https://kaggle.com/code/{meta['id']}", version=1, error=None)

    def kernel_status(self, ref: str) -> KernelStatus:
        return KernelStatus(status=self.status, failure_message="")

    def kernel_log(self, ref: str) -> str:
        return "x" * 100

    def download_output(self, ref: str, dest: str) -> list[str]:
        out = Path(dest) / "result.txt"
        out.write_text("42")
        return [str(out)]

    def delete_kernel(self, ref: str) -> None:
        self.kernels.remove(ref)

    def create_dataset(self, folder: str) -> str | None:
        meta = json.loads((Path(folder) / "dataset-metadata.json").read_text())
        self.datasets[meta["id"]] = sorted(p.name for p in Path(folder).iterdir())
        return None

    def dataset_status(self, ref: str) -> str:
        return self.dataset_states.pop(0) if self.dataset_states else "ready"

    def delete_dataset(self, ref: str) -> None:
        del self.datasets[ref]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    (p / "pkg").mkdir(parents=True)
    (p / "main.py").write_text("print('hi')")
    (p / "pkg" / "__init__.py").write_text("")
    (p / ".venv" / "lib").mkdir(parents=True)
    (p / ".venv" / "lib" / "big.so").write_bytes(b"x" * 10)
    (p / "__pycache__").mkdir()
    (p / "data").mkdir()
    (p / "data" / "train.csv").write_text("a,b\n1,2\n")
    return p


def make_jobs(tmp_path: Path, client: FakeClient) -> Jobs:
    return Jobs(
        client,
        Store(tmp_path / "home"),
        sleep=lambda _s: None,
        now=lambda: datetime(2026, 11, 15, 12, 0, 0),
    )


def zip_names(data: bytes) -> list[str]:
    return sorted(zipfile.ZipFile(BytesIO(data)).namelist())


def test_bundle_excludes_venv_caches_and_data_dir(project: Path) -> None:
    names = zip_names(bundle.build_code_zip(project, exclude=project / "data"))
    assert names == ["main.py", "pkg/__init__.py"]


def test_bundle_rejects_oversized_code(project: Path) -> None:
    import os

    (project / "weights.bin").write_bytes(os.urandom(bundle.MAX_BUNDLE_BYTES + 1))
    with pytest.raises(ValueError, match=r"weights\.bin"):
        bundle.build_code_zip(project)


def test_bootstrap_runs_entry_with_data_dir_env(tmp_path: Path, project: Path) -> None:
    """Execute the generated script locally with Kaggle paths remapped to tmp dirs."""
    (project / "pkg" / "util.py").write_text("VALUE = 7\n")
    (project / "main.py").write_text(
        "import os, pkg.util\n"
        "data = open(os.path.join(os.environ['KJOB_DATA_DIR'], 'train.csv')).read()\n"
        "open('out.txt', 'w').write(f'{pkg.util.VALUE} {len(data)} {__name__}')\n"
    )
    root = tmp_path / "fake_kaggle"
    (root / "working").mkdir(parents=True)
    mount = root / "input" / "datasets" / "me" / "kjobd-x"
    mount.mkdir(parents=True)
    (mount / "train.csv").write_text("a,b\n1,2\n")
    script = bundle.render_bootstrap(
        bundle.build_code_zip(project, exclude=project / "data"), "main.py", "kjobd-x"
    )
    script = script.replace("/kaggle/", f"{root}/").replace(
        "/tmp/kjob_code", str(tmp_path / "code")
    )
    path = tmp_path / "boot.py"
    path.write_text(script)
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True)
    assert (root / "working" / "out.txt").read_text() == "7 8 __main__"


def test_submit_with_data_dir_creates_dataset_and_attaches_it(
    tmp_path: Path, project: Path
) -> None:
    client = FakeClient()
    job = make_jobs(tmp_path, client).submit(
        str(project), "main.py", data_dir=str(project / "data"), gpu=True
    )
    assert job.id == "kjob-proj-20261115-120000"
    assert job.own_dataset == "me/kjobd-proj-20261115-120000"
    assert client.datasets[job.own_dataset] == ["dataset-metadata.json", "train.csv"]
    meta = client.pushed[0]
    assert meta["id"] == "me/kjob-proj-20261115-120000"
    assert meta["dataset_sources"] == [job.own_dataset]
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert bundle.slugify(str(meta["title"])) == str(meta["id"]).split("/")[1]


def test_submit_push_failure_removes_uploaded_dataset(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    client.push_error = "boom"
    with pytest.raises(RuntimeError, match="boom"):
        make_jobs(tmp_path, client).submit(str(project), "main.py", data_dir=str(project / "data"))
    assert client.datasets == {}
    assert Store(tmp_path / "home").jobs() == []


def test_jobs_survive_restart(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    job = make_jobs(tmp_path, client).submit(str(project), "main.py")
    client.status = "complete"
    restarted = make_jobs(tmp_path, client)  # new process: same store directory
    assert [j.id for j in restarted.list()] == [job.id]
    assert restarted.store.get_job(job.id).status == "complete"


def test_output_refuses_unfinished_job(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    job = jobs.submit(str(project), "main.py")
    with pytest.raises(ValueError, match="still running"):
        jobs.output(job.id)
    with pytest.raises(ValueError, match="refusing"):
        jobs.cleanup(job.id)


def test_output_of_ephemeral_job_downloads_then_cleans_kaggle(
    tmp_path: Path, project: Path
) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    job = jobs.submit(str(project), "main.py", data_dir=str(project / "data"))
    client.status = "complete"
    done, files = jobs.output(job.id)
    assert files == ["result.txt"]
    assert (project / "kaggle_out" / job.id / "result.txt").read_text() == "42"
    assert done.cleaned and client.kernels == set() and client.datasets == {}


def test_output_of_kept_job_leaves_kaggle_untouched(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    job = jobs.submit(str(project), "main.py", ephemeral=False)
    client.status = "error"
    done, _ = jobs.output(job.id)
    assert not done.cleaned and client.kernels == {job.kernel_ref}


def test_delete_dataset_guards(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    with pytest.raises(ValueError, match="not uploaded by kaggle-mcp"):
        jobs.delete_dataset("someone/public-data")
    ds = jobs.upload_dataset(str(project / "data"), "shared")
    jobs.submit(str(project), "main.py", datasets=[ds.ref])
    with pytest.raises(ValueError, match="unfinished jobs"):
        jobs.delete_dataset(ds.ref)
    client.status = "complete"
    jobs.list()
    jobs.delete_dataset(ds.ref)
    assert client.datasets == {} and jobs.store.datasets() == []


def _stored_job(job_id: str, **fields: Any) -> Job:
    return Job(
        id=job_id,
        kernel_ref=f"me/{job_id}",
        url=f"https://kaggle.com/code/me/{job_id}",
        project_dir="/p",
        entry="main.py",
        created_at="2026-11-15T12:00:00",
        **fields,
    )


def test_pending_lists_only_jobs_needing_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(tmp_path)
    store.put_job(_stored_job("running", status="running"))
    store.put_job(_stored_job("finished", status="complete"))
    store.put_job(_stored_job("downloaded", status="complete", output_dir="/out", cleaned=True))
    store.put_job(_stored_job("dropped", status="error", cleaned=True))
    monkeypatch.setenv("KAGGLE_MCP_HOME", str(tmp_path))
    assert cli.main(["pending"]) == 0
    out = capsys.readouterr().out
    assert "kjob wait" in out
    listed = [line.split()[0] for line in out.splitlines()[1:]]
    assert sorted(listed) == ["finished", "running"]


def _hammer_store(home: str, i: int) -> str:
    """Repeatedly write one job into a shared store; return the error, if any."""
    store = Store(Path(home))
    job = _stored_job(f"job{i}")
    try:
        for n in range(100):
            job.status = f"running-{n}"
            store.put_job(job)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


def test_concurrent_writers_keep_every_job(tmp_path: Path) -> None:
    # several `kjob wait` processes and the MCP server write the same store
    with ProcessPoolExecutor(4) as pool:
        errors = list(pool.map(_hammer_store, [str(tmp_path)] * 4, range(4)))
    assert errors == [""] * 4
    jobs = {j.id: j.status for j in Store(tmp_path).jobs()}
    assert jobs == {f"job{i}": "running-99" for i in range(4)}


def test_pending_prints_nothing_when_no_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KAGGLE_MCP_HOME", str(tmp_path))
    assert cli.main(["pending"]) == 0
    assert capsys.readouterr().out == ""
