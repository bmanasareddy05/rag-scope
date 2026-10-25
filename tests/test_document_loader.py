
from pathlib import Path

import pytest

from src.document_loader import load_pdf


def test_missing_pdf_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_file_raises_value_error(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("This is a test.", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(file_path)
