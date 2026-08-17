"""Safety regression tests for canonical archive extraction."""

import sys
from pathlib import Path
from zipfile import ZipInfo

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_fiqa import _validate_member  # noqa: E402


@pytest.mark.parametrize("member_name", ["../escape.txt", "fiqa/../../escape.txt", "/absolute.txt"])
def test_zip_path_traversal_is_rejected(tmp_path: Path, member_name: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe ZIP member path"):
        _validate_member(ZipInfo(member_name), tmp_path)


def test_normal_zip_member_is_accepted(tmp_path: Path) -> None:
    _validate_member(ZipInfo("fiqa/qrels/test.tsv"), tmp_path)
