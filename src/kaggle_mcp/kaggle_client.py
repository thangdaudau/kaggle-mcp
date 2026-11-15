"""The only module that touches the untyped `kaggle` package.

Every call runs with stdout redirected to stderr: the kaggle library prints
progress messages, and on an MCP stdio server stdout is the protocol channel.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import requests


@dataclass(frozen=True)
class PushResult:
    url: str
    version: int | None
    error: str | None


@dataclass(frozen=True)
class KernelStatus:
    status: (
        str  # queued | running | complete | error | cancel_requested | cancel_acknowledged | ...
    )
    failure_message: str


class KaggleClient(Protocol):
    def username(self) -> str: ...
    def push_kernel(self, folder: str) -> PushResult: ...
    def kernel_status(self, ref: str) -> KernelStatus: ...
    def kernel_log(self, ref: str) -> str: ...
    def download_output(self, ref: str, dest: str) -> list[str]: ...
    def delete_kernel(self, ref: str) -> None: ...
    def create_dataset(self, folder: str) -> str | None: ...
    def dataset_status(self, ref: str) -> str: ...
    def delete_dataset(self, ref: str) -> None: ...


@contextlib.contextmanager
def _stdout_to_stderr() -> Generator[None]:
    with contextlib.redirect_stdout(sys.stderr):
        yield


class RealKaggleClient:
    def __init__(self) -> None:
        with _stdout_to_stderr():
            from kaggle.api.kaggle_api_extended import (  # pyright: ignore[reportMissingTypeStubs] -- kaggle ships no type info
                KaggleApi,
            )

            api: Any = KaggleApi()
            api.authenticate()
        self._api: Any = api

    def username(self) -> str:
        name = self._api.get_config_value("username")
        if not name:
            raise RuntimeError("Kaggle username not found in config; check ~/.kaggle credentials")
        return str(name)

    def push_kernel(self, folder: str) -> PushResult:
        with _stdout_to_stderr():
            resp: Any = self._api.kernels_push(folder)
        error = resp.error or None
        invalid = list(resp.invalid_dataset_sources or [])
        if not error and invalid:
            error = f"invalid dataset sources: {invalid}"
        version = resp.version_number
        return PushResult(
            url=str(resp.url or ""),
            version=int(version) if version else None,
            error=str(error) if error else None,
        )

    def kernel_status(self, ref: str) -> KernelStatus:
        with _stdout_to_stderr():
            resp: Any = self._api.kernels_status(ref)
        status = resp.status
        name = status.name if hasattr(status, "name") else str(status)
        return KernelStatus(
            status=str(name).lower(), failure_message=str(resp.failure_message or "")
        )

    def kernel_log(self, ref: str) -> str:
        with _stdout_to_stderr():
            raw = str(self._api.kernels_logs(ref) or "")
        return _format_log(raw)

    def download_output(self, ref: str, dest: str) -> list[str]:
        with _stdout_to_stderr():
            files, _token = self._api.kernels_output(ref, dest, force=True, quiet=True)
        return [str(f) for f in files]

    def delete_kernel(self, ref: str) -> None:
        with _stdout_to_stderr():
            self._api.kernels_delete(ref, no_confirm=True)

    def create_dataset(self, folder: str) -> str | None:
        with _stdout_to_stderr():
            resp: Any = self._api.dataset_create_new(
                folder, public=False, quiet=True, dir_mode="zip"
            )
        error = getattr(resp, "error", None)
        return str(error) if error else None

    def dataset_status(self, ref: str) -> str:
        """A freshly created dataset answers 403/404 for a few seconds; report that as pending."""
        try:
            with _stdout_to_stderr():
                return str(self._api.dataset_status(ref)).lower()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 404):
                return "pending"
            raise

    def delete_dataset(self, ref: str) -> None:
        owner, slug = ref.split("/", 1)
        with _stdout_to_stderr():
            self._api.dataset_delete(owner, slug, no_confirm=True)


def _format_log(raw: str) -> str:
    """Kaggle returns the log as a JSON list of {stream_name, time, data}; flatten it."""
    try:
        entries: object = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(entries, list):
        return raw
    parts: list[str] = []
    for entry in cast(list[object], entries):
        if isinstance(entry, dict):
            parts.append(str(cast(dict[str, object], entry).get("data", "")))
    return "".join(parts)
