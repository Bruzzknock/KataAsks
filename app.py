import os
from typing import List

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

from utils import DATA_DIR, load_local_documents

load_dotenv()

st.set_page_config(page_title="Gemini Knowledge Chat", page_icon=":speech_balloon:")
st.title("Gemini Knowledge Chat")
st.caption("Chat with your local documents using a Gemini-powered RAG pipeline.")


@st.cache_resource(show_spinner="Loading models and documents...")
def build_graph(google_api_key: str, embedding_api_key: str):
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["RULEMAKER_API_KEY_GOOGLE"] = embedding_api_key

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=embedding_api_key
    )
    vector_store = Chroma(collection_name="streamlit_rag", embedding_function=embeddings)

    docs = load_local_documents(str(DATA_DIR))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if splits:
        _ = vector_store.add_documents(splits)

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
    return graph


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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources", expanded=False):
                render_sources(message["sources"])


if prompt := st.chat_input("Ask a question about your documents"):
    if not ready:
        st.warning("Provide both API keys in the sidebar to start chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            graph = build_graph(google_api_key, embedding_api_key)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = graph.invoke({"question": prompt})
                answer = result["answer"]
                st.markdown(answer)
                sources = result.get("context", [])
                if sources:
                    with st.expander("Sources", expanded=False):
                        render_sources(sources)
        except Exception as exc:
            answer = f"Could not run the chat: {exc}"
            sources = []
            st.error(answer)
        finally:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
