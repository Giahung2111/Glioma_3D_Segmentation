"""Background dedicated-GPU telemetry using NVML with nvidia-smi fallback."""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUSnapshot:
    timestamp: str
    elapsed_seconds: float
    gpu_name: str
    gpu_utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_w: float | None


@dataclass(frozen=True)
class GPUSummary:
    samples: int
    peak_memory_used_mb: float | None
    dedicated_memory_total_mb: float | None
    mean_gpu_utilization_percent: float | None
    peak_temperature_c: float | None
    mean_power_w: float | None
    backend: str
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _NVMLSampler:
    name = "nvml"

    def __init__(self, gpu_index: int) -> None:
        import pynvml  # type: ignore[import-not-found]

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def sample(self, elapsed: float) -> GPUSnapshot:
        nvml = self._pynvml
        memory = nvml.nvmlDeviceGetMemoryInfo(self._handle)
        utilization = nvml.nvmlDeviceGetUtilizationRates(self._handle)
        raw_name = nvml.nvmlDeviceGetName(self._handle)
        name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        try:
            power = nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except nvml.NVMLError:
            power = None
        return GPUSnapshot(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_seconds=elapsed,
            gpu_name=name,
            gpu_utilization_percent=float(utilization.gpu),
            memory_used_mb=memory.used / (1024**2),
            memory_total_mb=memory.total / (1024**2),
            temperature_c=float(
                nvml.nvmlDeviceGetTemperature(self._handle, nvml.NVML_TEMPERATURE_GPU)
            ),
            power_w=power,
        )

    def close(self) -> None:
        self._pynvml.nvmlShutdown()


class _NvidiaSMISampler:
    name = "nvidia-smi"
    _FIELDS = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"

    def __init__(self, gpu_index: int) -> None:
        self._gpu_index = gpu_index

    def sample(self, elapsed: float) -> GPUSnapshot:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self._gpu_index}",
                f"--query-gpu={self._FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
        )
        parts = [part.strip() for part in completed.stdout.strip().split(",")]
        if len(parts) != 6:
            raise RuntimeError(f"Unexpected nvidia-smi telemetry row: {completed.stdout!r}")
        power = None if parts[5] in {"[N/A]", "N/A", ""} else float(parts[5])
        return GPUSnapshot(
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_seconds=elapsed,
            gpu_name=parts[0],
            gpu_utilization_percent=float(parts[1]),
            memory_used_mb=float(parts[2]),
            memory_total_mb=float(parts[3]),
            temperature_c=float(parts[4]),
            power_w=power,
        )

    def close(self) -> None:
        return None


class GPUMonitor:
    """Sample one physical GPU without including Windows shared GPU memory."""

    _CSV_FIELDS = tuple(GPUSnapshot.__dataclass_fields__)

    def __init__(
        self,
        csv_path: Path,
        *,
        interval_seconds: float = 2.0,
        gpu_index: int = 0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.csv_path = csv_path.resolve()
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self._sampler: _NVMLSampler | _NvidiaSMISampler | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_monotonic = 0.0
        self._lock = threading.Lock()
        self._samples: list[GPUSnapshot] = []
        self._errors: list[str] = []

    @property
    def backend_name(self) -> str:
        return self._sampler.name if self._sampler else "not-started"

    @property
    def latest(self) -> GPUSnapshot | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def _create_sampler(self) -> _NVMLSampler | _NvidiaSMISampler:
        try:
            return _NVMLSampler(self.gpu_index)
        except Exception as exc:  # NVML may be unavailable even when nvidia-smi works.
            self._errors.append(f"NVML unavailable: {type(exc).__name__}: {exc}")
            sampler = _NvidiaSMISampler(self.gpu_index)
            sampler.sample(0.0)  # fail before launching the long job
            return sampler

    def start(self) -> GPUMonitor:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("GPU monitor is already running")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._sampler = self._create_sampler()
        self._start_monotonic = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="glioma-gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self._sampler is not None
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._CSV_FIELDS)
            writer.writeheader()
            handle.flush()
            while not self._stop_event.is_set():
                elapsed = time.monotonic() - self._start_monotonic
                try:
                    snapshot = self._sampler.sample(elapsed)
                    with self._lock:
                        self._samples.append(snapshot)
                    writer.writerow(asdict(snapshot))
                    handle.flush()
                except Exception as exc:
                    self._errors.append(f"{type(exc).__name__}: {exc}")
                self._stop_event.wait(self.interval_seconds)

    def stop(self) -> GPUSummary:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(10.0, self.interval_seconds * 3))
        if self._sampler:
            try:
                self._sampler.close()
            except Exception as exc:
                self._errors.append(f"sampler close: {type(exc).__name__}: {exc}")
        return self.summary()

    def summary(self) -> GPUSummary:
        with self._lock:
            samples = list(self._samples)
        powers = [sample.power_w for sample in samples if sample.power_w is not None]
        return GPUSummary(
            samples=len(samples),
            peak_memory_used_mb=max((sample.memory_used_mb for sample in samples), default=None),
            dedicated_memory_total_mb=max(
                (sample.memory_total_mb for sample in samples), default=None
            ),
            mean_gpu_utilization_percent=(
                sum(sample.gpu_utilization_percent for sample in samples) / len(samples)
                if samples
                else None
            ),
            peak_temperature_c=max((sample.temperature_c for sample in samples), default=None),
            mean_power_w=sum(powers) / len(powers) if powers else None,
            backend=self.backend_name,
            errors=tuple(self._errors),
        )

    def __enter__(self) -> GPUMonitor:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


def query_gpu_once(gpu_index: int = 0) -> GPUSnapshot:
    """Read a dedicated-memory snapshot for system checks/readiness gates."""

    sampler: _NVMLSampler | _NvidiaSMISampler
    try:
        sampler = _NVMLSampler(gpu_index)
    except Exception:
        sampler = _NvidiaSMISampler(gpu_index)
    try:
        return sampler.sample(0.0)
    finally:
        sampler.close()
