"""Safe subprocess execution with a live console/file tee."""

from __future__ import annotations

import datetime as dt
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

LineCallback = Callable[[str, str], None]
HeartbeatCallback = Callable[[], str | None]


@dataclass(frozen=True)
class LiveCommandResult:
    argv: tuple[str, ...]
    returncode: int
    started_at: str
    ended_at: str
    elapsed_seconds: float
    log_path: Path
    stdout_tail: tuple[str, ...]
    stderr_tail: tuple[str, ...]


class LiveCommandError(RuntimeError):
    """A failed command with concise, actionable diagnostics."""

    def __init__(self, result: LiveCommandResult) -> None:
        self.result = result
        diagnostic_lines = result.stderr_tail or result.stdout_tail
        relevant = "\n".join(diagnostic_lines[-20:]) or "(no process output)"
        super().__init__(
            f"Command failed with exit code {result.returncode}:\n"
            f"  {format_command(result.argv)}\n"
            f"Likely cause: {infer_likely_root_cause(diagnostic_lines)}\n"
            f"Last relevant output:\n{relevant}\n"
            f"Full log: {result.log_path}"
        )


def format_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(part) for part in argv])
    return shlex.join(str(part) for part in argv)


def infer_likely_root_cause(lines: Sequence[str]) -> str:
    text = "\n".join(lines).lower()
    if "out of memory" in text or "cuda error: out of memory" in text:
        return "GPU memory exhaustion; inspect patch/batch plan and other GPU processes."
    if "no space left" in text or "disk full" in text:
        return "Insufficient free disk space."
    if "cuda" in text and ("not available" in text or "driver" in text):
        return "CUDA/PyTorch/driver availability mismatch."
    if "filenotfounderror" in text or "is not recognized" in text:
        return "A required executable or input path was not found."
    if "dataset" in text and ("integrity" in text or "missing" in text):
        return "Dataset conversion or integrity validation failed."
    if "keyboardinterrupt" in text or "ctrl_c_event" in text:
        return "The process was interrupted; use the explicit resume option."
    return "Not identifiable from the final lines; inspect the full stage log."


def _reader(
    stream: TextIO,
    stream_name: str,
    output_queue: queue.Queue[tuple[str, str | None]],
) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put((stream_name, line.rstrip("\r\n")))
    finally:
        stream.close()
        output_queue.put((stream_name, None))


def _interrupt_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def run_live_command(
    argv: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stage: str = "PROCESS",
    line_callback: LineCallback | None = None,
    heartbeat_callback: HeartbeatCallback | None = None,
    heartbeat_interval_seconds: float = 20.0,
    check: bool = True,
    tail_lines: int = 80,
) -> LiveCommandResult:
    """Run argv without a shell while streaming stdout/stderr to console and log."""

    normalized = tuple(str(part) for part in argv)
    if not normalized:
        raise ValueError("argv cannot be empty")
    log_path = log_path.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    process_env.setdefault("PYTHONUNBUFFERED", "1")

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started_wall = dt.datetime.now(dt.timezone.utc)
    started_monotonic = time.monotonic()
    print(f"[{stage}] Command: {format_command(normalized)}", flush=True)
    print(f"[{stage}] Log: {log_path}", flush=True)

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_tail: deque[str] = deque(maxlen=tail_lines)
    stderr_tail: deque[str] = deque(maxlen=tail_lines)
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write(f"[{started_wall.isoformat()}] COMMAND {format_command(normalized)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            normalized,
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(
                target=_reader,
                args=(process.stdout, "stdout", output_queue),
                daemon=True,
            ),
            threading.Thread(
                target=_reader,
                args=(process.stderr, "stderr", output_queue),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        closed_streams = 0
        last_heartbeat = started_monotonic
        try:
            while closed_streams < 2 or process.poll() is None:
                try:
                    stream_name, line = output_queue.get(timeout=0.25)
                except queue.Empty:
                    stream_name, line = "", ""
                if line is None:
                    closed_streams += 1
                elif stream_name:
                    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                    log_handle.write(f"[{timestamp}] [{stream_name}] {line}\n")
                    log_handle.flush()
                    target = sys.stderr if stream_name == "stderr" else sys.stdout
                    print(line, file=target, flush=True)
                    (stderr_tail if stream_name == "stderr" else stdout_tail).append(line)
                    if line_callback:
                        line_callback(stream_name, line)

                now = time.monotonic()
                if heartbeat_callback and now - last_heartbeat >= heartbeat_interval_seconds:
                    heartbeat = heartbeat_callback()
                    if heartbeat:
                        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                        print(heartbeat, flush=True)
                        log_handle.write(f"[{timestamp}] [status] {heartbeat}\n")
                        log_handle.flush()
                    last_heartbeat = now
        except KeyboardInterrupt:
            print(
                f"[{stage}] Interrupt received; requesting graceful nnU-Net shutdown...",
                flush=True,
            )
            _interrupt_process(process)
            raise
        finally:
            for thread in threads:
                thread.join(timeout=2)

        returncode = process.wait()
        ended_wall = dt.datetime.now(dt.timezone.utc)
        elapsed = time.monotonic() - started_monotonic
        log_handle.write(
            f"[{ended_wall.isoformat()}] EXIT {returncode} elapsed_seconds={elapsed:.3f}\n"
        )

    result = LiveCommandResult(
        argv=normalized,
        returncode=returncode,
        started_at=started_wall.isoformat(),
        ended_at=ended_wall.isoformat(),
        elapsed_seconds=elapsed,
        log_path=log_path,
        stdout_tail=tuple(stdout_tail),
        stderr_tail=tuple(stderr_tail),
    )
    if check and result.returncode != 0:
        raise LiveCommandError(result)
    return result
