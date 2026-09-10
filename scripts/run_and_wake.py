"""Run one command in the background and queue one completion message.

The launcher deliberately has no retry loop, file watcher, or Codex polling.
The parent process creates the job directory and exits after starting a detached
supervisor.  The supervisor waits for the child with the OS process wait and
then makes exactly one ``codex queue`` attempt, including when the child could
not be started.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


QUEUE_TIMEOUT_SECONDS = 60
STATUS_SCHEMA = "run-and-wake-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_text(path: Path, text: str) -> None:
    """Write *text* and replace *path* atomically on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_status(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("status.json must contain an object")
    return value


def _write_status(path: Path, status: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    current = dict(status)
    current.update(updates)
    _atomic_json(path, current)
    return current


def _artifact_paths(job_dir: Path) -> dict[str, Path]:
    return {
        "status": job_dir / "status.json",
        "stdout": job_dir / "stdout.log",
        "stderr": job_dir / "stderr.log",
        "queue": job_dir / "queue.log",
        "supervisor": job_dir / "supervisor.log",
    }


def _completion_message(job_dir: Path, state: str, exit_code: int | None) -> str:
    """Return trusted, bounded wake text containing only artifact paths."""

    paths = _artifact_paths(job_dir.resolve())
    code = "unknown" if exit_code is None else str(exit_code)
    return (
        f"Background process finished with state={state}, exit_code={code}. "
        "Continue the already-authorized review/report workflow after inspecting "
        "the saved artifacts.\n"
        f"status: {paths['status']}\n"
        f"stdout: {paths['stdout']}\n"
        f"stderr: {paths['stderr']}\n"
        f"queue: {paths['queue']}\n"
        f"supervisor: {paths['supervisor']}"
    )


def _queue_argv(
    codex: str, thread: str, message: str, cwd: Path
) -> list[str]:
    """Build the native queue command without shell interpretation."""

    return [
        codex,
        "queue",
        "--thread",
        thread,
        "--message",
        message,
        "-C",
        str(cwd),
    ]


def _queue_once(
    *,
    codex: str,
    thread: str,
    message: str,
    cwd: Path,
    queue_log: Path,
    timeout_seconds: int = QUEUE_TIMEOUT_SECONDS,
    run=subprocess.run,
) -> dict[str, Any]:
    """Make one queue attempt and classify its transport result."""

    argv = _queue_argv(codex, thread, message, cwd)
    try:
        result = run(
            argv,
            cwd=str(cwd),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log = f"queue timed out after {timeout_seconds} seconds\n{output}"
        _atomic_text(queue_log, log)
        return {
            "delivery_status": "unknown",
            "queue_attempts": 1,
            "queue_exit_code": None,
            "queue_error": f"timeout after {timeout_seconds} seconds",
        }
    except OSError as exc:
        log = f"queue launch failed: {type(exc).__name__}: {exc}\n"
        _atomic_text(queue_log, log)
        return {
            "delivery_status": "failed",
            "queue_attempts": 1,
            "queue_exit_code": None,
            "queue_error": f"{type(exc).__name__}: {exc}",
        }

    output = result.stdout or ""
    _atomic_text(queue_log, output)
    if result.returncode == 0:
        return {
            "delivery_status": "queued",
            "queue_attempts": 1,
            "queue_exit_code": 0,
            "queue_error": None,
        }
    return {
        "delivery_status": "failed",
        "queue_attempts": 1,
        "queue_exit_code": result.returncode,
        "queue_error": f"codex queue exited with code {result.returncode}",
    }


def _supervise(
    *,
    job_dir: Path,
    thread: str,
    codex: str,
    cwd: Path,
    command: Sequence[str],
) -> int:
    paths = _artifact_paths(job_dir)
    status_path = paths["status"]
    status = _read_status(status_path)
    status = _write_status(
        status_path,
        status,
        state="running",
        supervisor_pid=os.getpid(),
        running_at=_utc_now(),
    )

    exit_code: int | None = None
    state = "failed"
    launch_error: str | None = None
    try:
        # DETACHED_PROCESS keeps the child independent of the launching console;
        # CREATE_NEW_PROCESS_GROUP prevents Ctrl+C propagation on Windows.
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            )
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdin": subprocess.DEVNULL,
            "stdout": paths["stdout"].open("w", encoding="utf-8", newline=""),
            "stderr": paths["stderr"].open("w", encoding="utf-8", newline=""),
            "shell": False,
            "close_fds": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True
        try:
            child = subprocess.Popen(list(command), **popen_kwargs)
        except BaseException:
            # The log handles were opened above and must be closed on a failed
            # Popen, otherwise Windows keeps the log files locked.
            popen_kwargs["stdout"].close()
            popen_kwargs["stderr"].close()
            raise
        status = _write_status(status_path, status, child_pid=child.pid)
        exit_code = child.wait()
        state = "completed" if exit_code == 0 else "failed"
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
        state = "failed"
        exit_code = None

    status = _write_status(
        status_path,
        status,
        state=state,
        exit_code=exit_code,
        launch_error=launch_error,
        finished_at=_utc_now(),
        delivery_status="pending",
        queue_attempts=0,
    )
    delivery = _queue_once(
        codex=codex,
        thread=thread,
        message=_completion_message(job_dir, state, exit_code),
        cwd=cwd,
        queue_log=paths["queue"],
    )
    _write_status(status_path, status, **delivery)
    return 0 if state == "completed" and exit_code == 0 else 1


def _detached_supervisor(
    *,
    job_dir: Path,
    thread: str,
    codex: str,
    cwd: Path,
    command: Sequence[str],
) -> int:
    paths = _artifact_paths(job_dir)
    supervisor_log = paths["supervisor"]
    with supervisor_log.open("a", encoding="utf-8", newline="") as log:
        # The supervisor's stdout/stderr were redirected by the parent. Keep
        # this handler for a final diagnostic if an unexpected error escapes.
        try:
            return _supervise(
                job_dir=job_dir,
                thread=thread,
                codex=codex,
                cwd=cwd,
                command=command,
            )
        except BaseException as exc:
            log.write(f"supervisor failed: {type(exc).__name__}: {exc}\n")
            log.flush()
            try:
                status = _read_status(paths["status"])
                _write_status(
                    paths["status"],
                    status,
                    state="failed",
                    exit_code=None,
                    launch_error=f"{type(exc).__name__}: {exc}",
                    finished_at=_utc_now(),
                    delivery_status="failed",
                    queue_attempts=0,
                )
            except BaseException as status_exc:
                log.write(
                    f"status update failed: {type(status_exc).__name__}: {status_exc}\n"
                )
                log.flush()
            return 1


def _create_start_status(
    *, job_dir: Path, thread: str, codex: str, cwd: Path, command: Sequence[str]
) -> dict[str, Any]:
    paths = _artifact_paths(job_dir)
    status = {
        "schema": STATUS_SCHEMA,
        "state": "start",
        "job_dir": str(job_dir),
        "cwd": str(cwd),
        "thread": thread,
        "codex": codex,
        "command": list(command),
        "supervisor_pid": None,
        "child_pid": None,
        "exit_code": None,
        "launch_error": None,
        "delivery_status": "not_started",
        "queue_attempts": 0,
        "queue_exit_code": None,
        "queue_error": None,
        "started_at": _utc_now(),
        "running_at": None,
        "finished_at": None,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _atomic_json(paths["status"], status)
    return status


def _launch(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir).expanduser().resolve()
    if job_dir.exists():
        raise SystemExit(f"--job-dir must not exist: {job_dir}")
    if not args.command:
        raise SystemExit("a command is required after --")
    if args.command[0] == "--":
        command = args.command[1:]
    else:
        command = args.command
    if not command:
        raise SystemExit("a command is required after --")
    thread = args.thread or os.environ.get("CODEX_THREAD_ID")
    if not thread:
        raise SystemExit("--thread or CODEX_THREAD_ID is required")
    cwd = Path.cwd().resolve()
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir()
    _create_start_status(
        job_dir=job_dir,
        thread=thread,
        codex=args.codex,
        cwd=cwd,
        command=command,
    )
    supervisor_log = _artifact_paths(job_dir)["supervisor"]
    supervisor_log_handle = supervisor_log.open("a", encoding="utf-8", newline="")
    supervisor_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_supervise",
        "--job-dir",
        str(job_dir),
        "--thread",
        thread,
        "--codex",
        args.codex,
        "--cwd",
        str(cwd),
        "--",
        *command,
    ]
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": supervisor_log_handle,
        "stderr": subprocess.STDOUT,
        "shell": False,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        supervisor = subprocess.Popen(supervisor_argv, **popen_kwargs)
    except BaseException:
        supervisor_log_handle.close()
        raise
    supervisor_log_handle.close()
    # Do not rewrite status here: a fast supervisor may already have advanced
    # it to running/completed, and a stale parent write could roll that state
    # back.  The supervisor records its own PID as its first atomic update.
    print(f"detached supervisor pid={supervisor.pid} job_dir={job_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--thread")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--_supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cwd", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._supervise:
        if not args.cwd or not args.thread:
            raise SystemExit("internal supervisor arguments are incomplete")
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            raise SystemExit("internal supervisor command is empty")
        return _detached_supervisor(
            job_dir=Path(args.job_dir).resolve(),
            thread=args.thread,
            codex=args.codex,
            cwd=Path(args.cwd).resolve(),
            command=command,
        )
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
