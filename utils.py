from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "documents"     # always correct relative to this script
VECTOR_STORE_DIR = BASE_DIR / "chroma_store"

print(f"[info] Looking for documents under: {DATA_DIR}")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Folder not found: {DATA_DIR}")


def load_local_documents(data_dir: str) -> list[Document]:
    """Recursively load docs from a folder using format-appropriate loaders."""
    data_path = Path(data_dir)
    docs: list[Document] = []

    for path in data_path.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()

        try:
            if suffix in {".txt"}:
                loader = TextLoader(str(path), encoding="utf-8")
            elif suffix in {".md"}:
                loader = UnstructuredMarkdownLoader(str(path))
            elif suffix in {".pdf"}:
                loader = PyPDFLoader(str(path))  # or PyMuPDFLoader if you prefer
            elif suffix in {".docx"}:
                loader = Docx2txtLoader(str(path))
            else:
                # skip unknown file types; add more loaders as needed
                continue

            loaded_docs = loader.load()
            stat = path.stat()
            source_path = str(path)
            for doc in loaded_docs:
                doc.metadata["source"] = source_path
                doc.metadata["source_name"] = path.name
                doc.metadata["source_mtime"] = stat.st_mtime
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"[warn] Skipping {path.name}: {e}")

    return docs
