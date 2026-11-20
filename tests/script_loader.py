"""Helper for loading standalone scripts as importable modules in tests.

Scripts under ``scripts/`` live outside the installed ``sampleworks`` package, so they
can't be imported normally. ``load_script`` imports one by filesystem path instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_script(script_path: Path) -> ModuleType:
    """Import a standalone ``.py`` script by path so tests don't require it on ``sys.path``.

    Parameters
    ----------
    script_path : Path
        Filesystem path to the script to import.

    Returns
    -------
    ModuleType
        The imported module, with the script's top-level names available as attributes.
    """
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
