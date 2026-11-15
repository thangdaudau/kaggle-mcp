# kaggle-mcp

MCP server that runs local Python projects as background jobs on Kaggle's free machines (12 h per run, CPU or GPU).

## Setup

```bash
# credentials: token from kaggle.com/settings/api → "Generate New Token"
mkdir -p ~/.kaggle && chmod 700 ~/.kaggle   # save token to ~/.kaggle/access_token, chmod 600

uv tool install --editable .                # installs `kaggle-mcp` and `kjob` to ~/.local/bin
claude mcp add --scope user kaggle-jobs -- ~/.local/bin/kaggle-mcp
```

## How a job runs

1. `submit_job(project_dir, entry, data_dir=None, datasets=[], gpu=False, internet=True, ephemeral=True)`
   zips the project (without `.git`, virtualenvs, caches, `kaggle_out/` and `data_dir`), embeds it in a
   generated script and pushes it as a private Kaggle script kernel. `data_dir` is uploaded as a private
   dataset first. The call returns as soon as Kaggle accepts the job.
2. On Kaggle the code is unpacked to `/tmp/kjob_code`, `requirements.txt` is pip-installed if present,
   then `entry` runs as `__main__` with cwd `/kaggle/working`.
3. `job_output` downloads everything the job wrote to `/kaggle/working` (max ~20 GB) plus the log to
   `<project_dir>/kaggle_out/<job_id>/`. For `ephemeral` jobs it then deletes the kernel and the per-job
   dataset from Kaggle.

Environment variables available to the job:

| Variable | Value |
|---|---|
| `KJOB_DATA_DIR` | mount of `data_dir` (read-only); subdirectories keep their structure |
| `KJOB_CODE_DIR` | unpacked project code |
| `KJOB_DEADLINE` | epoch seconds when the run is stopped (`time_limit_h`, default 11.5 h) |
| `KJOB_RESUME_DIR` | output of the `resume_from` job (read-only), if given |

### Runs longer than 12 hours

Kaggle stops every run after 12 hours. A job is stopped at `KJOB_DEADLINE` instead, before that
cutoff, so everything it wrote to the working directory is saved; `job_output` then reports
`incomplete: true` and keeps the kernel on Kaggle. Submit the same job again with
`resume_from=<job_id>` to continue: the previous output is mounted at `KJOB_RESUME_DIR`, without a
round trip through the local machine. Once a resumed part is downloaded, the part it continued from
is deleted from Kaggle.

```python
import json, os
from pathlib import Path

resume = os.environ.get("KJOB_RESUME_DIR")
state = json.loads((Path(resume) / "ckpt.json").read_text()) if resume else {"step": 0}
while state["step"] < TOTAL_STEPS:
    train_one_step(state)
    state["step"] += 1
    Path("ckpt.json").write_text(json.dumps(state))  # cwd is the job's output
```

A script that stops early on its own (for example by checking `KJOB_DEADLINE`) should create the
file `kjob_incomplete` in its cwd so `job_output` reports it as incomplete. For a chain of parts,
upload the input data once with `upload_dataset` and pass `datasets=[...]` rather than `data_dir`.

## Tools

| Tool | Purpose |
|---|---|
| `submit_job` | start a job |
| `job_status`, `list_jobs` | status: `queued`, `running`, `complete`, `error`, `cancel_*` |
| `job_logs` | stdout/stderr tail (published after the run ends) |
| `job_output` | download outputs; cleans Kaggle for ephemeral jobs |
| `cleanup_job` | delete a finished job's kernel and per-job dataset |
| `upload_dataset`, `list_datasets`, `delete_dataset` | data shared across jobs; delete refuses while an unfinished job uses it |

## Waiting for a job

```bash
kjob wait <job_id>   # polls every 60 s, prints status changes, exit 0 = complete, 1 = failed
kjob list
kjob pending         # jobs not downloaded yet or stopped and not resumed; local store only (SessionStart hook)
```

Job records live in `~/.local/share/kaggle-mcp/jobs.json`, so jobs remain visible after a restart.

## Development

```bash
uv run pytest && uv run ruff check . && uv run pyright
```
