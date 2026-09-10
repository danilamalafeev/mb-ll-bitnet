"""Device, reproducibility and honest timing helpers. No implicit CPU fallback."""
import os
import platform
import random
import resource
import sys

import torch


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this Python environment")
    if device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("supported devices: cpu, mps, cuda")
    return device


def seed_everything(seed, deterministic=True, cpu_threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(deterministic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    # MPS uses unified memory; do not label it dedicated VRAM or invent a peak.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and KiB on Linux.
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 1024
    return {
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "process_peak_rss_mb": rss_mb,
    }


def environment(device=None):
    result = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
        "device": str(device) if device is not None else None,
    }
    if device is not None and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(device_name=props.name, total_vram_mb=props.total_memory / 2**20)
    return result


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
