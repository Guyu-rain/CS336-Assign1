"""Minimal compatibility shim for the Unix-only ``resource`` module on Windows.

This test suite imports ``resource`` unconditionally, but only exercises
``RLIMIT_AS``, ``getrlimit``, and ``setrlimit`` on Linux via skip-guarded tests.
On Windows we provide no-op implementations so import-time collection succeeds.
"""

from __future__ import annotations

RLIMIT_AS = 9

_limits: dict[int, tuple[int, int]] = {RLIMIT_AS: (-1, -1)}


def getrlimit(resource: int) -> tuple[int, int]:
    return _limits.get(resource, (-1, -1))


def setrlimit(resource: int, limits: tuple[int, int]) -> None:
    _limits[resource] = limits
