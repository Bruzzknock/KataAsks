from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "documents"     # always correct relative to this script
VECTOR_STORE_DIR = BASE_DIR / "chroma_store"

print(f"[info] Looking for documents under: {DATA_DIR}")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {DATA_DIR}")


def _has_text(documents: Iterable[Document]) -> bool:
    for doc in documents:
        if (doc.page_content or "").strip():
            return True
    return False


def _load_pdf_with_ocr(path: Path) -> list[Document]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install pypdfium2 to enable OCR fallback for PDFs."
        ) from exc

    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install pytesseract to enable OCR fallback for PDFs."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install Pillow to enable OCR fallback for PDFs."
        ) from exc

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:  # pragma: no cover - env specific
        raise RuntimeError(
            "Tesseract executable not found. Update TESSERACT_CMD or install Tesseract."
        ) from exc

    stat = path.stat()
    source_path = str(path)

    pdf = pdfium.PdfDocument(str(path))
    ocr_docs: list[Document] = []

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            try:
                bitmap = page.render(scale=300 / 72)
                try:
                    pil_image = bitmap.to_pil()
                finally:
                    bitmap.close()
            finally:
                page.close()

            if not isinstance(pil_image, Image.Image):
                raise RuntimeError("pypdfium2 did not return a PIL image")
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            text = pytesseract.image_to_string(pil_image).strip()
            if not text:
                continue

            metadata = {
                "source": source_path,
                "source_name": path.name,
                "source_mtime": stat.st_mtime,
                "page": page_index,
                "page_label": str(page_index + 1),
                "processing": "ocr",
                "ocr_engine": "tesseract",
            }
            ocr_docs.append(Document(page_content=text, metadata=metadata))
    finally:
        pdf.close()

    if not ocr_docs:
        raise RuntimeError("OCR produced no extractable text")

    return ocr_docs


def _load_pdf_with_fallback(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    if _has_text(docs):
        return docs

    print(f"[info] Running OCR fallback for {path.name} (no extractable text detected)")
    return _load_pdf_with_ocr(path)


def load_local_documents(data_dir: str, skip_filenames: Iterable[str] | None = None) -> list[Document]:
    """Recursively load docs from a folder using format-appropriate loaders.

    Optionally skip files whose names already exist in the vector store.
    """
    data_path = Path(data_dir)
    skip_set = {name.lower() for name in (skip_filenames or []) if name}
    docs: list[Document] = []

    for path in data_path.rglob("*"):
        if not path.is_file():
            continue

        if skip_set and path.name.lower() in skip_set:
            print(f"[info] Skipping already embedded file before load: {path.name}")
            continue

        suffix = path.suffix.lower()

        try:
            if suffix in {".txt"}:
                loaded_docs = TextLoader(str(path), encoding="utf-8").load()
            elif suffix in {".md"}:
                loaded_docs = UnstructuredMarkdownLoader(str(path)).load()
            elif suffix in {".pdf"}:
                loaded_docs = _load_pdf_with_fallback(path)
            elif suffix in {".docx"}:
                loaded_docs = Docx2txtLoader(str(path)).load()
            else:
                # skip unknown file types; add more loaders as needed
                continue

            stat = path.stat()
            source_path = str(path)
            for doc in loaded_docs:
                metadata = doc.metadata or {}
                metadata["source"] = source_path
                metadata["source_name"] = path.name
                metadata["source_mtime"] = stat.st_mtime

                if "page_label" not in metadata:
                    page_value = metadata.get("page")
                    if isinstance(page_value, (int, float)):
                        metadata["page_label"] = str(int(page_value) + 1)
                    elif isinstance(page_value, str) and page_value.strip():
                        metadata["page_label"] = page_value.strip()

                doc.metadata = metadata
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"[warn] Skipping {path.name}: {e}")

    return docs
