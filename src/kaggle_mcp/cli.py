"""`kjob` command: wait for or list Kaggle jobs from the shell."""

from __future__ import annotations

import argparse
import sys

from kaggle_mcp.jobs import Jobs
from kaggle_mcp.kaggle_client import RealKaggleClient
from kaggle_mcp.store import Job, Store


def _line(job: Job) -> str:
    extra = f" ({job.failure_message})" if job.failure_message else ""
    return f"{job.id}  {job.status}{extra}  {job.url}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kjob")
    sub = parser.add_subparsers(dest="cmd", required=True)
    wait = sub.add_parser(
        "wait", help="block until a job finishes; exit 0 if complete, 1 otherwise"
    )
    wait.add_argument("job_id")
    wait.add_argument("--poll", type=float, default=60, help="seconds between checks")
    sub.add_parser("list", help="list jobs with current status")
    sub.add_parser(
        "pending",
        help="list jobs not downloaded yet or stopped and not resumed (local store only)",
    )
    args = parser.parse_args(argv)

    if args.cmd == "pending":
        all_jobs = Store().jobs()
        pending = [j for j in all_jobs if not j.cleaned and j.output_dir is None]
        resumed = {j.resume_from for j in all_jobs}
        stalled = [
            j for j in all_jobs if j.incomplete and not j.cleaned and j.kernel_ref not in resumed
        ]
        if pending:
            print(
                "Unfinished Kaggle jobs (kaggle-jobs MCP). For each: run `kjob wait <id>` "
                "in the background, then call job_output when it exits."
            )
            for job in pending:
                print(_line(job))
        if stalled:
            print(
                "Kaggle jobs stopped at their time limit and not resumed yet. For each: "
                "submit_job(..., resume_from=<id>) to continue, or cleanup_job(<id>) to drop it."
            )
            for job in stalled:
                print(_line(job))
        return 0

    jobs = Jobs(RealKaggleClient(), Store())
    if args.cmd == "list":
        for job in jobs.list():
            print(_line(job))
        return 0

    job = jobs.wait(args.job_id, poll_s=args.poll, on_change=lambda j: print(_line(j), flush=True))
    return 0 if job.status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
