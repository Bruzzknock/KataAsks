import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from utils import DATA_DIR, VECTOR_STORE_DIR, load_local_documents

load_dotenv()

st.set_page_config(page_title="Kata's Knowledge Chat", page_icon=":speech_balloon:")
st.title("Kata's Knowledge Chat")
st.caption("Find architecture solutions using Kata's Knowledge Chat RAG pipeline.")


def format_source_label(source: str) -> str:
    try:
        relative = Path(source).resolve().relative_to(DATA_DIR.resolve())
        return relative.as_posix()
    except Exception:
        return Path(source).name


def format_timestamp(mtime: float | None) -> str:
    if not mtime:
        return "unknown"
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def generate_chunk_id(source: str, mtime: float | None, index: int) -> str:
    source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
    mtime_component = "na" if mtime is None else str(int(mtime * 1000))
    return f"{source_hash}-{mtime_component}-{index}"

def persist_vector_store(vector_store: Chroma) -> None:
    """Flush vector store state when persistence is available."""
    if hasattr(vector_store, "persist"):
        persist_vector_store(vector_store)
        return
    client = getattr(vector_store, "_client", None)
    if client and hasattr(client, "persist"):
        client.persist()




def sync_local_documents(vector_store: Chroma) -> dict[str, Any]:
    try:
        docs = load_local_documents(str(DATA_DIR))
    except FileNotFoundError as exc:
        return {"added": 0, "updated": 0, "skipped": 0, "error": str(exc)}

    if not docs:
        return {"added": 0, "updated": 0, "skipped": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in splits:
        source = chunk.metadata.get("source")
        if not source:
            continue
        entry = grouped.setdefault(
            source,
            {
                "mtime": chunk.metadata.get("source_mtime"),
                "name": chunk.metadata.get("source_name") or Path(source).name,
                "chunks": [],
            },
        )
        entry["chunks"].append(chunk)

    added = updated = skipped = 0
    modified = False

    for source, entry in grouped.items():
        mtime = entry["mtime"]
        existing = vector_store.get(where={"source": source}, include=["metadatas"])
        ids = existing.get("ids", []) if existing else []
        metadatas = existing.get("metadatas", []) if existing else []
        existing_mtimes = {
            md.get("source_mtime")
            for md in metadatas
            if md and md.get("source_mtime") is not None
        }
        same_mtime = existing_mtimes == {mtime} if existing_mtimes else False
        same_chunk_count = len(ids) == len(entry["chunks"]) if ids else False

        if ids and same_mtime and same_chunk_count:
            skipped += 1
            continue

        if ids:
            vector_store.delete(where={"source": source})
            updated += 1
            modified = True
        else:
            added += 1

        doc_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
        new_ids: List[str] = []
        for idx, chunk in enumerate(entry["chunks"]):
            chunk.metadata["source"] = source
            chunk.metadata["source_name"] = entry["name"]
            chunk.metadata["source_mtime"] = mtime
            chunk.metadata["doc_id"] = doc_hash
            chunk.metadata["chunk_index"] = idx
            new_ids.append(generate_chunk_id(source, mtime, idx))

        vector_store.add_documents(entry["chunks"], ids=new_ids)
        modified = True

    if modified:
        persist_vector_store(vector_store)

    return {"added": added, "updated": updated, "skipped": skipped}


def get_persisted_documents(vector_store: Chroma) -> List[dict[str, Any]]:
    collection = vector_store.get(include=["metadatas"])
    metadatas = collection.get("metadatas", []) if collection else []
    aggregated: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not metadata:
            continue
        source = metadata.get("source")
        if not source:
            continue
        entry = aggregated.setdefault(
            source,
            {
                "name": metadata.get("source_name") or Path(source).name,
                "mtime": metadata.get("source_mtime"),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1

    results: List[dict[str, Any]] = []
    for source, data in aggregated.items():
        results.append(
            {
                "source": source,
                "name": data["name"],
                "mtime": data["mtime"],
                "chunks": data["chunks"],
            }
        )

    results.sort(key=lambda item: item["name"].lower())
    return results


def delete_document(vector_store: Chroma, source: str) -> int:
    existing = vector_store.get(where={"source": source})
    ids = existing.get("ids", []) if existing else []
    if not ids:
        return 0
    vector_store.delete(where={"source": source})
    persist_vector_store(vector_store)
    return len(ids)


def render_document_panel(vector_store: Chroma) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Embedded documents")

    message = st.session_state.pop("doc_action_message", None)
    if message:
        level = message.get("level", "info")
        text = message.get("text", "")
        if level == "success":
            st.sidebar.success(text)
        elif level == "error":
            st.sidebar.error(text)
        else:
            st.sidebar.info(text)

    if st.sidebar.button("Sync local folder", key="sync_docs"):
        summary = sync_local_documents(vector_store)
        if summary.get("error"):
            st.session_state["doc_action_message"] = {
                "level": "error",
                "text": summary["error"],
            }
        else:
            added = summary.get("added", 0)
            updated = summary.get("updated", 0)
            if added or updated:
                st.session_state["doc_action_message"] = {
                    "level": "success",
                    "text": f"Indexed {added + updated} document(s): {added} new, {updated} updated.",
                }
            else:
                st.session_state["doc_action_message"] = {
                    "level": "info",
                    "text": "No new or updated documents detected.",
                }
        st.experimental_rerun()

    docs = get_persisted_documents(vector_store)
    if not docs:
        st.sidebar.caption("No embedded documents yet. Use Sync local folder to index files.")
        st.session_state.setdefault("doc_selection", None)
        return

    for doc in docs:
        display_name = format_source_label(doc["source"])
        timestamp = format_timestamp(doc.get("mtime"))
        st.sidebar.markdown(
            f"- **{display_name}**  \n  {doc['chunks']} chunk(s) - updated {timestamp}"
        )

    options = [None] + [doc["source"] for doc in docs]
    selection = st.sidebar.selectbox(
        "Delete a document",
        options=options,
        index=0,
        format_func=lambda src: "Select a document" if src is None else format_source_label(src),
        key="doc_selection",
    )

    if st.sidebar.button("Delete selected document", key="delete_doc", disabled=selection is None):
        removed = delete_document(vector_store, selection) if selection else 0
        if removed:
            text = f"Removed {format_source_label(selection)} ({removed} chunk(s))."
            level = "success"
        else:
            text = "Nothing was removed; the document may have already been deleted."
            level = "info"
        st.session_state["doc_action_message"] = {"level": level, "text": text}
        st.session_state["doc_selection"] = None
        st.experimental_rerun()


@st.cache_resource(show_spinner="Loading models and documents...")
def build_graph(google_api_key: str, embedding_api_key: str):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["RULEMAKER_API_KEY_GOOGLE"] = embedding_api_key

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=embedding_api_key
    )
    vector_store = Chroma(
        collection_name="streamlit_rag",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    sync_local_documents(vector_store)

    prompt = hub.pull("rlm/rag-prompt")

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State)
    graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph, vector_store


def ensure_configuration():
    with st.sidebar:
        st.header("Configuration")
        st.markdown(f"Using documents from `{DATA_DIR}`.")
        default_llm_key = os.environ.get("GOOGLE_API_KEY", "")
        default_embed_key = os.environ.get("RULEMAKER_API_KEY_GOOGLE", "")

        google_api_key = st.text_input(
            "Google Gemini API key", value=default_llm_key, type="password"
        )
        embedding_api_key = st.text_input(
            "Embedding API key", value=default_embed_key, type="password"
        )
        ready = bool(google_api_key and embedding_api_key)

        if not ready:
            st.info("Enter both API keys to enable the chat.")

        return google_api_key, embedding_api_key, ready


def render_sources(docs: List[Document]):
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("file_path") or f"Document {idx}"
        text = doc.page_content.strip()
        snippet = text[:500] + ("..." if len(text) > 500 else "")
        st.markdown(f"**{source}**\n\n> {snippet}")


google_api_key, embedding_api_key, ready = ensure_configuration()

if "messages" not in st.session_state:
    st.session_state.messages = []

graph = None
vector_store = None
initialization_error: str | None = None

if ready:
    try:
        graph, vector_store = build_graph(google_api_key, embedding_api_key)
    except Exception as exc:  # noqa: BLE001
        initialization_error = str(exc)
        st.error(f"Could not initialize the chat pipeline: {exc}")

if vector_store is not None:
    render_document_panel(vector_store)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources", expanded=False):
                render_sources(message["sources"])

if prompt := st.chat_input("Ask a question about your documents"):
    if not ready:
        st.warning("Provide both API keys in the sidebar to start chatting.")
    elif graph is None:
        if initialization_error:
            st.error(initialization_error)
        else:
            st.error("The chat pipeline is not ready yet.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = graph.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)
                sources = result.get("context", [])
                if sources:
                    with st.expander("Sources", expanded=False):
                        render_sources(sources)
        except Exception as exc:  # noqa: BLE001
            answer = f"Could not run the chat: {exc}"
            sources = []
            st.error(answer)
        finally:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
