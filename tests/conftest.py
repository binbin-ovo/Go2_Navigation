"""Keep optional native LightNav imports local to their test module."""

import importlib
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def isolated_native_lightnav():
    source = Path(os.environ["LIGHTNAV_ASSETS"]) / "source-local-v2/src"

    def native_names():
        return [
            name
            for name in sys.modules
            if name == "lightnav" or name.startswith("lightnav.") or name == "lightnav_worker"
        ]

    previous = {name: sys.modules.pop(name) for name in native_names()}
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.syspath_prepend(str(source))
            policy = importlib.import_module("lightnav.inference.policies")
            assert Path(policy.__file__).resolve().is_relative_to(source.resolve())
            yield
    finally:
        for name in native_names():
            sys.modules.pop(name)
        sys.modules.update(previous)
