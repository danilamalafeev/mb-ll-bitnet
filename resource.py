"""Small Windows compatibility surface for Python's absent resource module."""

from ctypes import POINTER, Structure, WinDLL, byref, c_size_t, sizeof, wintypes

RUSAGE_SELF = 0


class _Usage(Structure):
    _fields_ = [("ru_maxrss", c_size_t)]


class _ProcessMemoryCounters(Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", c_size_t),
        ("WorkingSetSize", c_size_t),
        ("QuotaPeakPagedPoolUsage", c_size_t),
        ("QuotaPagedPoolUsage", c_size_t),
        ("QuotaPeakNonPagedPoolUsage", c_size_t),
        ("QuotaNonPagedPoolUsage", c_size_t),
        ("PagefileUsage", c_size_t),
        ("PeakPagefileUsage", c_size_t),
    ]


def getrusage(_who):
    counters = _ProcessMemoryCounters()
    counters.cb = sizeof(counters)
    kernel32 = WinDLL("kernel32", use_last_error=True)
    process = kernel32.GetCurrentProcess()
    psapi = WinDLL("psapi", use_last_error=True)
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [wintypes.HANDLE, POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    get_process_memory_info.restype = wintypes.BOOL
    ok = get_process_memory_info(process, byref(counters), counters.cb)
    usage = _Usage()
    # runtime.py interprets non-macOS ru_maxrss as KiB.
    usage.ru_maxrss = counters.PeakWorkingSetSize // 1024 if ok else 0
    return usage
