
from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str | Path) -> list[dict]:
    """Extract non-empty PDF pages and preserve source and page metadata."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        if text:
            pages.append({
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_number,
                },
            })

    return pages
