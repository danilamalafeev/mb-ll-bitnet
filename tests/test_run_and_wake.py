from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_and_wake


def test_queue_argv_uses_exact_list_and_no_shell(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class Result:
        returncode = 0
        stdout = "queued message id=abc\n"

    def fake_run(argv: list[str], **kwargs: object) -> Result:
        calls.append((argv, kwargs))
        return Result()

    message = "path with spaces & shell-looking text"
    outcome = run_and_wake._queue_once(
        codex="C:\\Program Files\\Codex\\codex.exe",
        thread="01a085b8-7f43-7d31-a17a-3835fe3e88da",
        message=message,
        cwd=tmp_path,
        queue_log=tmp_path / "queue.log",
        run=fake_run,
    )

    assert calls == [
        (
            [
                "C:\\Program Files\\Codex\\codex.exe",
                "queue",
                "--thread",
                "01a085b8-7f43-7d31-a17a-3835fe3e88da",
                "--message",
                message,
                "-C",
                str(tmp_path),
            ],
            {
                "cwd": str(tmp_path),
                "shell": False,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "timeout": 60,
                "check": False,
            },
        )
    ]
    assert outcome == {
        "delivery_status": "queued",
        "queue_attempts": 1,
        "queue_exit_code": 0,
        "queue_error": None,
    }
    assert (tmp_path / "queue.log").read_text(encoding="utf-8") == "queued message id=abc\n"


def test_queue_nonzero_is_failed_with_one_attempt(tmp_path: Path) -> None:
    calls = 0

    class Result:
        returncode = 23
        stdout = "queue rejected\n"

    def fake_run(*args: object, **kwargs: object) -> Result:
        nonlocal calls
        calls += 1
        return Result()

    outcome = run_and_wake._queue_once(
        codex="codex",
        thread="thread",
        message="wake",
        cwd=tmp_path,
        queue_log=tmp_path / "queue.log",
        run=fake_run,
    )
    assert calls == 1
    assert outcome["delivery_status"] == "failed"
    assert outcome["queue_attempts"] == 1
    assert outcome["queue_exit_code"] == 23


def test_queue_timeout_is_unknown_without_retry(tmp_path: Path) -> None:
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(kwargs["timeout"], "queue", output="accepted?\n")

    outcome = run_and_wake._queue_once(
        codex="codex",
        thread="thread",
        message="wake",
        cwd=tmp_path,
        queue_log=tmp_path / "queue.log",
        run=fake_run,
    )
    assert calls == 1
    assert outcome == {
        "delivery_status": "unknown",
        "queue_attempts": 1,
        "queue_exit_code": None,
        "queue_error": "timeout after 60 seconds",
    }
    assert "accepted?" in (tmp_path / "queue.log").read_text(encoding="utf-8")


def test_completion_message_contains_only_bounded_artifact_references(tmp_path: Path) -> None:
    message = run_and_wake._completion_message(tmp_path, "failed", None)
    assert "Continue the already-authorized review/report workflow" in message
    assert "status: " + str((tmp_path / "status.json").resolve()) in message
    assert "stdout: " + str((tmp_path / "stdout.log").resolve()) in message
    assert "stderr: " + str((tmp_path / "stderr.log").resolve()) in message
    assert "queue: " + str((tmp_path / "queue.log").resolve()) in message
    assert "supervisor: " + str((tmp_path / "supervisor.log").resolve()) in message
    assert "accepted?" not in message


def test_start_status_is_atomic_and_accounts_single_delivery_slot(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    run_and_wake._create_start_status(
        job_dir=job,
        thread="thread",
        codex="codex",
        cwd=tmp_path,
        command=["python", "-c", "print('x & y')"],
    )
    status = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert status["schema"] == "run-and-wake-v1"
    assert status["state"] == "start"
    assert status["command"] == ["python", "-c", "print('x & y')"]
    assert status["delivery_status"] == "not_started"
    assert status["queue_attempts"] == 0
    assert not list(job.glob("*.tmp"))


def test_launch_rejects_existing_job_directory(tmp_path: Path) -> None:
    job = tmp_path / "existing"
    job.mkdir()
    with pytest.raises(SystemExit, match="must not exist"):
        run_and_wake.main(
            [
                "--job-dir",
                str(job),
                "--thread",
                "thread",
                "--",
                "python",
                "-c",
                "pass",
            ]
        )
