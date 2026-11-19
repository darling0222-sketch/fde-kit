from __future__ import annotations

from pathlib import Path

import pytest

from fde_kit.config import Settings
from fde_kit.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def runtime(tmp_path_factory) -> Runtime:
    home = tmp_path_factory.mktemp("fde-home")
    settings = Settings(
        home=home / "data",
        corpus=ROOT / "examples" / "corpus",
        ontology=ROOT / "ontology" / "after_sales.yaml",
    )
    rt = Runtime.create(settings)
    rt.compile(reset=True)
    yield rt
    rt.close()


@pytest.fixture(scope="session")
def store(runtime: Runtime):
    return runtime.store


@pytest.fixture(scope="session")
def agent(runtime: Runtime):
    return runtime.agent


@pytest.fixture(scope="session")
def ontology(runtime: Runtime):
    return runtime.ontology
