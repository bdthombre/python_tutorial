"""Streamlit UI for a PDF based mini-RAG demo."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import streamlit as st

import api_key


@dataclass
class DocumentState:
    chunks: List[str]
    vectors: np.ndarray
    file_name: str


TEXT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
client = OpenAI(api_key=api_key.openai)


def read_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    file_buffer = io.BytesIO(uploaded_file.getvalue())
    reader = PdfReader(file_buffer)
    pages = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        cleaned = extracted.replace("\x00", " ").strip()
        if cleaned:
            pages.append(cleaned)
    return "\n\n".join(pages)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Obtain embedding vectors for the provided chunks."""
    response = client.embeddings.create(model=EMBED_MODEL, input=list(texts))
    return np.array([row.embedding for row in response.data], dtype=np.float32)


def top_chunks(query: str, chunks: Sequence[str], vectors: np.ndarray, k: int = 4) -> Tuple[List[str], np.ndarray]:
    """Return the top-k chunks that best match the query."""
    q_response = client.embeddings.create(model=EMBED_MODEL, input=[query])
    q_vec = np.array(q_response.data[0].embedding, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec) + 1e-10

    chunk_norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    normalized = vectors / chunk_norm
    sims = normalized @ q_vec

    order = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in order], sims[order]


def ask_model(question: str, context_chunks: Sequence[str]) -> str:
    """Use the OpenAI Responses API to answer with retrieved context."""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant that answers strictly from the context.

Context:
{context}

Question: {question}

If the answer is not present, reply with \"I don't know\"."""
    response = client.responses.create(
        model=TEXT_MODEL,
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text.strip()


def build_sidebar(state: DocumentState | None) -> None:
    st.sidebar.header("Steps")
    st.sidebar.markdown("""
1. Upload a PDF file.
2. Wait for chunking + embeddings.
3. Ask natural language questions.
4. Review the retrieved context to debug answers.
""")
    if state:
        st.sidebar.success(f"Loaded {len(state.chunks)} chunks from {state.file_name}.")
    else:
        st.sidebar.info("Upload a PDF to get started.")


def update_state(uploaded_file) -> DocumentState | None:
    text = read_pdf(uploaded_file)
    if not text.strip():
        st.warning("Could not find any extractable text inside that PDF.")
        return None
    with st.spinner("Chunking and creating embeddings..."):
        chunks = chunk_text(text)
        vectors = embed_texts(chunks)
    st.success(f"Created {len(chunks)} chunks and stored their embeddings.")
    return DocumentState(chunks=chunks, vectors=vectors, file_name=uploaded_file.name)


def main() -> None:
    st.set_page_config(page_title="PDF RAG chat", page_icon="📄", layout="wide")
    st.title("PDF RAG playground")
    st.write(
        "Upload a PDF, let the app build embeddings locally, and then chat with the "
        "document using the OpenAI Responses API."
    )

    if "doc_state" not in st.session_state:
        st.session_state.doc_state = None

    build_sidebar(st.session_state.doc_state)

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="visible")
    if uploaded is not None:
        st.session_state.doc_state = update_state(uploaded)

    query = st.text_area(
        "Ask a question about the document",
        placeholder="e.g. Summarize the third section...",
        height=120,
    )

    ask_disabled = st.session_state.doc_state is None or not query.strip()
    if st.button("Ask", type="primary", disabled=ask_disabled):
        state = st.session_state.doc_state
        if not state:
            st.info("Upload a PDF first.")
        else:
            top_chunks_list, similarities = top_chunks(query, state.chunks, state.vectors)
            answer = ask_model(query, top_chunks_list)
            st.subheader("Answer")
            st.write(answer)

            st.divider()
            st.subheader("Retrieved context")
            for idx, (chunk, score) in enumerate(zip(top_chunks_list, similarities), start=1):
                with st.expander(f"Chunk {idx} — similarity {score:.3f}", expanded=idx == 1):
                    st.write(chunk)


if __name__ == "__main__":
    main()
