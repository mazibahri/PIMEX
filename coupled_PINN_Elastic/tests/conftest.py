import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption("--data", default=os.path.join(ROOT, "data"), help="folder with the PLAXIS tables")
    parser.addoption("--model", default="Elastic", help="file name prefix of the PLAXIS tables")


@pytest.fixture
def plate_file(request):
    path = os.path.join(request.config.getoption("--data"),
                        f"{request.config.getoption('--model')}_Plate_forces_TablePlateForces_LinPhase.TXT")
    if not os.path.exists(path):
        pytest.skip(f"PLAXIS data not found ({path})")
    return path
