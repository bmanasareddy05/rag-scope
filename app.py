
import tempfile
from pathlib import Path

import streamlit as st

from src.chunker import chunk_documents
from src.document_loader import load_pdf
from src.generator import AnswerGenerator
from src.retriever import HybridRetriever


st.set_page_config(
    page_title="RAG-Scope",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 RAG-Scope")
st.caption(
    "Document intelligence powered by hybrid retrieval, "
    "local embeddings, and Groq."
)


@st.cache_resource(show_spinner="Reading, chunking, and indexing PDF...")
def build_pipeline(pdf_bytes: bytes, filename: str):
    """Build a retriever from an uploaded PDF."""

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = Path(temp_file.name)

        pages = load_pdf(temp_path)

        if not pages:
            raise ValueError(
                "No extractable text was found in this PDF."
            )

        chunks = chunk_documents(
            pages,
            chunk_size=900,
            overlap=150,
        )

        # Flatten metadata for the retriever and answer generator.
        prepared_chunks = []

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            prepared_chunks.append({
                "text": chunk["text"],
                "source": metadata.get("source", filename),
                "page": metadata.get("page", "Unknown"),
                "metadata": metadata,
            })

        if not prepared_chunks:
            raise ValueError(
                "No text chunks were created from this PDF."
            )

        retriever = HybridRetriever(prepared_chunks)

        return retriever, prepared_chunks, len(pages)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


with st.sidebar:
    st.header("1. Upload a document")

    uploaded_file = st.file_uploader(
        "Choose a text-based PDF",
        type=["pdf"],
    )

    st.caption(
        "Your PDF is processed locally for text extraction "
        "and retrieval. Relevant chunks are sent to Groq "
        "when you ask a question."
    )

    index_button = st.button(
        "Process document",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if index_button and uploaded_file is not None:
        # Clear the previous document's active index.
        st.session_state.pop("rag_pipeline", None)

        try:
            retriever, chunks, page_count = build_pipeline(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

            st.session_state["rag_pipeline"] = {
                "retriever": retriever,
                "chunks": chunks,
                "filename": uploaded_file.name,
                "page_count": page_count,
            }

            st.success("Document indexed successfully!")

        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")


pipeline = st.session_state.get("rag_pipeline")

if pipeline:
    st.success(
        f"Active document: {pipeline['filename']} · "
        f"{pipeline['page_count']} pages · "
        f"{len(pipeline['chunks'])} chunks"
    )

    st.header("2. Ask a question")

    question = st.text_input(
        "What would you like to know?",
        placeholder="Ask something based on your PDF...",
    )

    top_k = st.slider(
        "Number of context chunks to retrieve",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching and generating an answer..."):
                    retrieved_chunks = pipeline[
                        "retriever"
                    ].search(
                        question,
                        top_k=top_k,
                        candidate_k=15,
                    )

                    generator = AnswerGenerator()
                    result = generator.generate(
                        question,
                        retrieved_chunks,
                    )

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")

                citations = result.get("citations", [])
                valid_citations = [
                    index
                    for index in citations
                    if isinstance(index, int)
                    and not isinstance(index, bool)
                    and 0 <= index < len(retrieved_chunks)
                ]

                if valid_citations:
                    for index in valid_citations:
                        chunk = retrieved_chunks[index]

                        with st.expander(
                            f"{chunk.get('source', 'Unknown')} "
                            f"— Page {chunk.get('page', 'Unknown')}"
                        ):
                            st.write(chunk["text"])
                            st.caption(
                                f"Retrieval score: "
                                f"{chunk.get('score', 0.0):.4f}"
                            )
                else:
                    st.info(
                        "The answer did not include supporting "
                        "chunk citations. Check the retrieved "
                        "context below before relying on it."
                    )

                with st.expander("Inspect retrieved context"):
                    for index, chunk in enumerate(retrieved_chunks):
                        st.markdown(
                            f"**Chunk {index + 1}: "
                            f"{chunk.get('source', 'Unknown')} "
                            f"— Page {chunk.get('page', 'Unknown')}**"
                        )
                        st.write(chunk["text"])
                        st.divider()

            except Exception as exc:
                st.error(
                    f"Unable to generate the answer: {exc}"
                )

else:
    st.info(
        "Upload a PDF in the sidebar and select "
        "'Process document' to get started."
    )

st.divider()
st.caption(
    "RAG-Scope | PDF ingestion · Hybrid search · "
    "Grounded generation · Source inspection"
)
