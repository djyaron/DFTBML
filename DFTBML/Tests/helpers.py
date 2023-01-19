import os
import shutil
from pathlib import Path


def get_dftbplus_executable() -> str:
    dftbplus_exec_path = shutil.which("dftb+")

    if dftbplus_exec_path is None:
        dftbplus_exec_path = os.environ.get("DFTBPLUS_EXEC_PATH")

    if dftbplus_exec_path is None:
        raise ValueError("Could not find dftb+ executable")

    return dftbplus_exec_path


package_dir = Path(__file__).resolve().parents[1]

tests_dir = package_dir / "Tests"
test_data_dir = package_dir / "test_files"

ani1_path = test_data_dir / "ANI-1ccx_clean_fullentry.h5"
auorg_dir = package_dir / "Auorg_1_1/"
mio_dir = package_dir / "MIO_0_1/"
test_skf_dir = package_dir / "TestSKF/"
