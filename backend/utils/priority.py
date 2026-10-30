"""
Process / thread priority on Windows (ctypes, no extra dependency).

The vision cycle shares the PC with the HMI browser, an IDE and whatever
else runs on it. Measured on the test images: a first read that takes
~300 ms took 1.0-1.5 s while another program used 3-4 cores, and those
cycles made the HMI lag. Raising the process to ABOVE_NORMAL and the
vision thread to HIGHEST lets the scheduler run the read first; it does
not change what is read. Elsewhere (not Windows) these are no-ops.
"""

import sys

from backend.utils.logger import get_logger

log = get_logger(__name__)

PROCESS_CLASSES = {
    "normal": 0x00000020,          # NORMAL_PRIORITY_CLASS
    "above_normal": 0x00008000,    # ABOVE_NORMAL_PRIORITY_CLASS
    "high": 0x00000080,            # HIGH_PRIORITY_CLASS
}
THREAD_PRIORITIES = {
    "normal": 0,                   # THREAD_PRIORITY_NORMAL
    "above_normal": 1,             # THREAD_PRIORITY_ABOVE_NORMAL
    "highest": 2,                  # THREAD_PRIORITY_HIGHEST
}


_K32 = None


def _kernel32():
    """kernel32 with the signatures used here (HANDLE, not ctypes' default
    32-bit int: the pseudo-handles of GetCurrentProcess / GetCurrentThread
    are -1 / -2 and must reach the API as 64-bit handles). None when not
    on Windows."""
    global _K32
    if sys.platform != "win32":
        return None
    if _K32 is None:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentThread.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k32.GetPriorityClass.restype = wintypes.DWORD
        k32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
        k32.SetThreadPriority.restype = wintypes.BOOL
        k32.GetThreadPriority.argtypes = [wintypes.HANDLE]
        k32.GetThreadPriority.restype = ctypes.c_int
        _K32 = k32
    return _K32


def set_process_priority(name: str) -> bool:
    """Priority class of this process ("normal", "above_normal", "high");
    False when not set (not Windows, unknown name, or the call failed)."""
    k32, value = _kernel32(), PROCESS_CLASSES.get(str(name).lower())
    if k32 is None or value is None:
        return False
    ok = bool(k32.SetPriorityClass(k32.GetCurrentProcess(), value))
    if not ok:
        log.warning("SetPriorityClass(%s) failed", name)
    return ok


def set_thread_priority(name: str) -> bool:
    """Priority of the calling thread ("normal", "above_normal", "highest");
    False when not set."""
    k32, value = _kernel32(), THREAD_PRIORITIES.get(str(name).lower())
    if k32 is None or value is None:
        return False
    ok = bool(k32.SetThreadPriority(k32.GetCurrentThread(), value))
    if not ok:
        log.warning("SetThreadPriority(%s) failed", name)
    return ok
