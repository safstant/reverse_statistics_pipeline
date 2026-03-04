"""
check_normaliz.py — Step 0: Normaliz binary detection and environment setup.

Run standalone:  python check_normaliz.py
Import:          from .check_normaliz import check; found, path, version = check()

This module must be imported BEFORE any other reverse_stats module so that the
REVERSE_STATS_NORMALIZ_PATH environment variable is set before the per-module
_find_normaliz() calls fire their 'not found' warnings.

Search order (first match wins):
  1. REVERSE_STATS_NORMALIZ_PATH env var   (already set by previous call)
  2. NORMALIZ_PATH env var
  3. shutil.which('normaliz')              (anything on PATH)
  4. shutil.which('normaliz.exe')          (Windows PATH variant)
  5. C:\\normaliz\\normaliz.exe            (user's confirmed install location)
  6. C:\\Program Files\\Normaliz\\normaliz.exe
  7. C:\\Program Files (x86)\\Normaliz\\normaliz.exe
  8. /usr/bin/normaliz                     (Linux)
  9. /usr/local/bin/normaliz               (Linux local / macOS Homebrew)
 10. /opt/normaliz/bin/normaliz            (Linux opt)
 11. /opt/homebrew/bin/normaliz            (macOS Apple Silicon)

Confirmed binary: C:\\normaliz\\normaliz.exe  (Normaliz 3.2.0, 2025-10-01)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import platform
from typing import Tuple, Optional

# ── Candidate paths in priority order ────────────────────────────────────────

_WINDOWS_PATHS = [
    r"C:\normaliz\normaliz.exe",           # user's confirmed install (position 1)
    r"C:\Program Files\Normaliz\normaliz.exe",
    r"C:\Program Files (x86)\Normaliz\normaliz.exe",
]

_LINUX_PATHS = [
    "/usr/bin/normaliz",
    "/usr/local/bin/normaliz",
    "/opt/normaliz/bin/normaliz",
]

_MACOS_PATHS = [
    "/usr/local/bin/normaliz",
    "/opt/homebrew/bin/normaliz",
    "/opt/local/bin/normaliz",
]


def _platform_paths() -> list[str]:
    system = platform.system()
    if system == "Windows":
        return _WINDOWS_PATHS
    elif system == "Darwin":
        return _MACOS_PATHS
    else:
        return _LINUX_PATHS


def _candidates() -> list[Optional[str]]:
    return [
        os.environ.get("REVERSE_STATS_NORMALIZ_PATH"),  # 1 — already set
        os.environ.get("NORMALIZ_PATH"),                # 2 — alternative env var
        shutil.which("normaliz"),                       # 3 — PATH lookup
        shutil.which("normaliz.exe"),                   # 4 — Windows PATH
        *_platform_paths(),                             # 5-11 — platform dirs
    ]


def _try_binary(path: str) -> Optional[str]:
    """
    Attempt to run 'path --version'.  Returns the first line of stdout/stderr
    on success, None on any failure.
    """
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = (result.stdout or result.stderr or "").strip()
        # Normaliz --version writes to stderr on 3.x
        return output.split("\n")[0] if output else "unknown version"
    except Exception:
        return None


def check() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Probe for a working Normaliz binary.

    Returns
    -------
    (found, path, version)
        found   : True if a working binary was located
        path    : absolute path to the binary, or None
        version : first line of `normaliz --version`, or None
    """
    for candidate in _candidates():
        if not candidate:
            continue
        if not (os.path.isfile(candidate) and os.access(candidate, os.X_OK)):
            continue
        version = _try_binary(candidate)
        if version is not None:
            # Pin the path so all subsequent module imports use it
            os.environ["REVERSE_STATS_NORMALIZ_PATH"] = candidate
            return True, candidate, version

    return False, None, None


# ── Singleton result (cached after first call) ────────────────────────────────
_cached: Optional[Tuple[bool, Optional[str], Optional[str]]] = None


def get() -> Tuple[bool, Optional[str], Optional[str]]:
    """Cached version of check() — safe to call many times with no overhead."""
    global _cached
    if _cached is None:
        _cached = check()
    return _cached


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("reverse_stats — Step 0: Normaliz detection")
    print("=" * 60)
    found, path, version = check()
    if found:
        print(f"  Status  : OK")
        print(f"  Binary  : {path}")
        print(f"  Version : {version}")
        print(f"  Env var : REVERSE_STATS_NORMALIZ_PATH={path}")
        print()
        print("  Ready for V15.3 — Normaliz-backed vertex enumeration")
        print("  and triangulation will be active.")
    else:
        print("  Status  : NOT FOUND")
        print()
        print("  Searched:")
        for c in _candidates():
            if c:
                exists = os.path.isfile(c)
                print(f"    {'[X]' if exists else '[ ]'}  {c}")
        print()
        print("  To fix:")
        print("    Option A: Place normaliz.exe in C:\\normaliz\\")
        print("    Option B: Set env var REVERSE_STATS_NORMALIZ_PATH=/path/to/normaliz")
        print("    Option C: Put normaliz on your system PATH")
        print()
        print("  Pipeline will fall back to LLL-based triangulation.")
    print("=" * 60)
