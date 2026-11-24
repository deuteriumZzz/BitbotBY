"""Каждая точка входа из реестра находится и запускается через `python -m`."""

import importlib.util
from pathlib import Path

import pytest

from bitbot.config import entrypoints


@pytest.mark.parametrize("module", entrypoints.ALL)
def test_entrypoint_is_runnable(module):
    spec = importlib.util.find_spec(module)
    assert spec is not None, f"{module} не найден"
    if spec.submodule_search_locations:  # пакет: python -m нужен __main__.py
        assert importlib.util.find_spec(f"{module}.__main__"), f"{module} без __main__"
    else:
        src = Path(spec.origin).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__"' in src, f"{module} без блока __main__"


def test_python_m_and_package_root():
    assert entrypoints.python_m(entrypoints.API, "py") == ["py", "-m", "bitbot.api"]
    assert (Path(entrypoints.package_root()) / "bitbot" / "__init__.py").is_file()
