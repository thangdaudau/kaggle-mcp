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
        self.incomplete = False  # job output contains the "stopped at time limit" marker

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
        files = [str(out)]
        if self.incomplete:
            marker = Path(dest) / bundle.INCOMPLETE_MARKER
            marker.touch()
            files.append(str(marker))
        return files

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
        bundle.build_code_zip(project, exclude=project / "data"),
        "main.py",
        "kjobd-x",
        resume_slug=None,
        time_limit_s=60,
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


def _run_bootstrap(
    tmp_path: Path, project: Path, *, resume_slug: str | None = None, time_limit_s: int = 60
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Run the generated script locally with /kaggle/ remapped to tmp_path/fake_kaggle."""
    root = tmp_path / "fake_kaggle"
    (root / "working").mkdir(parents=True)
    (root / "input").mkdir(exist_ok=True)
    script = bundle.render_bootstrap(
        bundle.build_code_zip(project, exclude=project / "data"),
        "main.py",
        None,
        resume_slug=resume_slug,
        time_limit_s=time_limit_s,
    )
    script = script.replace("/kaggle/", f"{root}/").replace(
        "/tmp/kjob_code", str(tmp_path / "code")
    )
    path = tmp_path / "boot.py"
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=30)
    return root, proc


def test_bootstrap_stops_entry_at_time_limit_and_keeps_output(
    tmp_path: Path, project: Path
) -> None:
    (project / "main.py").write_text(
        "import time\n"
        "try:\n"
        "    open('ckpt.txt', 'w').write('step1')\n"
        "    time.sleep(20)\n"
        "    open('done.txt', 'w').write('x')\n"
        "finally:\n"
        "    open('finally.txt', 'w').write('ran')\n"
    )
    root, proc = _run_bootstrap(tmp_path, project, time_limit_s=1)
    assert proc.returncode == 0, proc.stderr
    work = root / "working"
    assert (work / bundle.INCOMPLETE_MARKER).exists()
    assert (work / "ckpt.txt").read_text() == "step1"
    assert (work / "finally.txt").read_text() == "ran"
    assert not (work / "done.txt").exists()


def test_bootstrap_sets_deadline_and_no_marker_when_entry_finishes(
    tmp_path: Path, project: Path
) -> None:
    (project / "main.py").write_text(
        "import os, time\n"
        "left = float(os.environ['KJOB_DEADLINE']) - time.time()\n"
        "open('left.txt', 'w').write(str(left))\n"
    )
    root, proc = _run_bootstrap(tmp_path, project, time_limit_s=60)
    assert proc.returncode == 0, proc.stderr
    assert 50 < float((root / "working" / "left.txt").read_text()) <= 60
    assert not (root / "working" / bundle.INCOMPLETE_MARKER).exists()


def test_bootstrap_exposes_previous_job_output_as_resume_dir(tmp_path: Path, project: Path) -> None:
    (project / "main.py").write_text(
        "import os\n"
        "ckpt = open(os.path.join(os.environ['KJOB_RESUME_DIR'], 'ckpt.txt')).read()\n"
        "open('resumed.txt', 'w').write(ckpt)\n"
    )
    prev = tmp_path / "fake_kaggle" / "input" / "notebooks" / "me" / "kjob-prev"
    prev.mkdir(parents=True)
    (prev / "ckpt.txt").write_text("step7")
    root, proc = _run_bootstrap(tmp_path, project, resume_slug="kjob-prev")
    assert proc.returncode == 0, proc.stderr
    assert (root / "working" / "resumed.txt").read_text() == "step7"


def test_submit_resume_from_attaches_previous_kernel_output(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    first = jobs.submit(str(project), "main.py", time_limit_h=0.5)
    with pytest.raises(ValueError, match="still running"):
        jobs.submit(str(project), "main.py", resume_from=first.id)
    client.status = "complete"
    second = jobs.submit(str(project), "main.py", resume_from=first.id)
    assert client.pushed[-1]["kernel_sources"] == [first.kernel_ref]
    assert second.resume_from == first.kernel_ref
    assert "RESUME_SLUG = 'kjob-proj-20261115-120000'" in client.scripts[-1]
    assert "TIME_LIMIT_S = 1800" in client.scripts[0]
    with pytest.raises(KeyError):
        jobs.submit(str(project), "main.py", resume_from="nope")


def test_submit_rejects_time_limit_outside_kaggle_range(tmp_path: Path, project: Path) -> None:
    jobs = make_jobs(tmp_path, FakeClient())
    for bad in (0, -1, 12.5):
        with pytest.raises(ValueError, match="time_limit_h"):
            jobs.submit(str(project), "main.py", time_limit_h=bad)


def test_incomplete_job_is_kept_until_its_resumer_is_downloaded(
    tmp_path: Path, project: Path
) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    first = jobs.submit(str(project), "main.py")
    client.status, client.incomplete = "complete", True
    done, _ = jobs.output(first.id)
    assert done.incomplete and not done.cleaned and first.kernel_ref in client.kernels

    jobs = Jobs(  # next part gets a distinct slug
        client, jobs.store, sleep=lambda _s: None, now=lambda: datetime(2026, 11, 15, 13, 0, 0)
    )
    client.status = "running"
    second = jobs.submit(str(project), "main.py", resume_from=first.id)
    with pytest.raises(ValueError, match="resum"):
        jobs.cleanup(first.id)
    client.status, client.incomplete = "complete", False
    final, _ = jobs.output(second.id)
    assert not final.incomplete and final.cleaned
    assert jobs.store.get_job(first.id).cleaned
    assert client.kernels == set()


def test_cleaning_a_cleaned_resume_source_is_skipped(tmp_path: Path, project: Path) -> None:
    client = FakeClient()
    jobs = make_jobs(tmp_path, client)
    first = jobs.submit(str(project), "main.py")
    client.status, client.incomplete = "complete", True
    jobs.output(first.id)
    jobs.cleanup(first.id)
    with pytest.raises(ValueError, match="cleaned"):
        jobs.submit(str(project), "main.py", resume_from=first.id)


def test_pending_lists_incomplete_jobs_not_yet_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(tmp_path)
    done = {"status": "complete", "output_dir": "/out", "incomplete": True}
    store.put_job(_stored_job("stalled", **done))
    store.put_job(_stored_job("continued", **done))
    store.put_job(_stored_job("next-part", status="running", resume_from="me/continued"))
    store.put_job(_stored_job("dropped", **done, cleaned=True))
    monkeypatch.setenv("KAGGLE_MCP_HOME", str(tmp_path))
    assert cli.main(["pending"]) == 0
    out = capsys.readouterr().out
    assert "resume_from" in out
    listed = sorted(line.split()[0] for line in out.splitlines() if "https://" in line)
    assert listed == ["next-part", "stalled"]


def test_pending_prints_nothing_when_no_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KAGGLE_MCP_HOME", str(tmp_path))
    assert cli.main(["pending"]) == 0
    assert capsys.readouterr().out == ""
