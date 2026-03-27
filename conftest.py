from __future__ import annotations

import builtins
import os
import tempfile
from pathlib import Path

import pytest
import _pytest.pathlib


_ROOT = Path(__file__).resolve().parent
_LOCAL_TMP = _ROOT / "results" / "tmp"
_SESSION_BASETEMP = _ROOT / "results" / "pytest_tmp" / str(os.getpid())
_LOCAL_TMP.mkdir(exist_ok=True)
_SESSION_BASETEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_LOCAL_TMP)
os.environ["TEMP"] = str(_LOCAL_TMP)
os.environ["TMPDIR"] = str(_LOCAL_TMP)

_original_open = builtins.open
_UTF8_SUFFIXES = {".txt", ".json"}


def _patched_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if "b" not in mode and encoding is None:
        path = Path(file) if isinstance(file, (str, os.PathLike)) else None
        if path is not None and path.suffix.lower() in _UTF8_SUFFIXES:
            try:
                resolved = path.resolve(strict=False)
            except OSError:
                resolved = path
            if _ROOT in resolved.parents or resolved == _ROOT:
                encoding = "utf-8"
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


builtins.open = _patched_open


def pytest_configure(config):
    config.option.basetemp = str(_SESSION_BASETEMP)
    config.option.cache_dir = str(_ROOT / ".pytest_cache_local")
    _pytest.pathlib.cleanup_dead_symlinks = lambda root: None


@pytest.fixture
def tmp_path() -> Path:
    return Path(tempfile.mkdtemp(dir=_LOCAL_TMP))
