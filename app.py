# app.py
import os
import io
import time
import math
from typing import List, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import numpy as np
import faiss
import openai

# ---------- Configuration ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable not found. Set it before running the app.")
    st.stop()
openai.api_key = OPENAI_API_KEY

EMBED_MODEL = "text-embedding-3-small"   # embeddings model (change if desired)
CHAT_MODEL = "gpt-4o"                    # chat model for summarization and QA (as in your Java sample)
EMBED_DIM = 1536                         # typical embedding dim for older OpenAI; adjust if your chosen model differs

# ---------- Utilities ----------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text_pages = []
        for p in range(len(reader.pages)):
            page = reader.pages[p]
            text_pages.append(page.extract_text() or "")
        return "\n".join(text_pages)
    except Exception as e:
        st.warning(f"PDF parsing failed: {e}")
        return ""

def fetch_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "SummarizerBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # remove scripts, styles, nav, footer
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # heuristics: join large <p> blocks
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        # fallback: take text from body
        if not paragraphs:
            body = soup.body.get_text(separator=" ", strip=True) if soup.body else soup.get_text(separator=" ", strip=True)
            return body

        # filter short paragraphs
        paragraphs = [p for p in paragraphs if len(p) > 60]
        return "\n\n".join(paragraphs)
    except Exception as e:
        st.warning(f"Unable to fetch URL: {e}")
        return ""

def chunk_text(text: str, chunk_size:int = 1000, overlap:int = 200) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk = " ".join(chunk_tokens)
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ---------- OpenAI / Embeddings / FAISS ----------
def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Call OpenAI embeddings for a list of texts. Returns numpy arrays."""
    vectors = []
    # batch requests in groups to avoid huge single calls
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # use openai.Embedding.create
        resp = openai.Embedding.create(model=EMBED_MODEL, input=batch)
        for item in resp["data"]:
            vectors.append(np.array(item["embedding"], dtype=np.float32))
    return vectors

def build_faiss_index(vectors: List[np.ndarray]) -> faiss.IndexFlatIP:
    if not vectors:
        return None
    dim = vectors[0].shape[0]
    # Normalize vectors for cosine similarity via inner product
    vecs = np.vstack(vectors)
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms==0] = 1
    vecs = vecs / norms
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    return index

def retrieve_top_k(query:str, chunks:List[str], index:faiss.IndexFlatIP, vectors:List[np.ndarray], top_k:int=5) -> List[Tuple[int,float,str]]:
    """Return list of (idx, score, text) for top_k retrieved chunks."""
    if index is None or len(chunks) == 0:
        return []
    q_vec = embed_texts([query])[0].astype(np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    q = q_vec.reshape(1, -1)
    D, I = index.search(q, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((int(idx), float(score), chunks[idx]))
    return results

# ---------- Generation ----------
def chat_completion(system_prompt: str, user_prompt: str, max_tokens: int=512) -> str:
    """Simple wrapper for chat completions using OpenAI Chat API."""
    # Keep messages concise
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # Use try-except for robustness
    try:
        resp = openai.ChatCompletion.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error from OpenAI ChatCompletion: {e}]"

def generate_summary_and_points(full_text: str, retriever_context: List[str]=None) -> Tuple[str,str]:
    """Produce a summary abstract and numbered key points (top 5)."""
    # Build a compact context to include if provided (few top chunks)
    context = ""
    if retriever_context:
        context = "\n\n---\n\n".join(retriever_context)
    system_prompt = (
        "You are a concise summarizer for technical material. "
        "Given the transcript or document text and optional retrieved context, produce:\n"
        "1) A short abstract paragraph (1-3 sentences) in direct, plain language, avoiding flowery phrasing.\n"
        "2) A numbered list of the top 5 key points, each as a single, direct sentence.\n"
        "Do not refer to the speaker or use meta language like 'the speaker said'."
    )
    user_prompt = "DOCUMENT TEXT:\n\n" + (context + "\n\n" + full_text if context else full_text)
    out = chat_completion(system_prompt, user_prompt, max_tokens=500)
    # Try to split abstract and points heuristically
    if "\n1" in out:
        abstract, points = out.split("\n1", 1)
        points = "1" + points
    else:
        # fallback: return whole output as abstract
        abstract = out
        points = ""
    return abstract.strip(), points.strip()

def answer_question_with_context(question: str, retrieved_chunks: List[Tuple[int,float,str]]) -> str:
    # Compose context block with chunk indices and content
    context_parts = []
    for idx, score, chunk in retrieved_chunks:
        context_parts.append(f"### CHUNK {idx} (score={score:.3f})\n{chunk}")
    context_text = "\n\n".join(context_parts)
    system_prompt = (
        "You are a helpful, concise assistant that answers questions using only the provided document context. "
        "When the answer is not contained in the context, say 'I don't know based on the provided document.' "
        "Base your response on context, and cite the chunk indices you used in square brackets, e.g., [CHUNK 2]."
    )
    user_prompt = f"CONTEXT:\n\n{context_text}\n\nQUESTION: {question}\n\nAnswer concisely and cite chunk indices you used."
    return chat_completion(system_prompt, user_prompt, max_tokens=400)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Text Summarizer", page_icon="✂️", layout="wide")

st.markdown("""
    <style>
    .stApp { font-family: 'Inter', sans-serif; }
    .header { display:flex; align-items:center; gap:12px; }
    .brand { font-size:20px; font-weight:700; }
    .muted { color: #6c757d; }
    .source-box { background:#f8f9fa; padding:12px; border-radius:8px; }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("<div class='header'><div class='brand'>RAG Text Summarizer</div><div class='muted'>— Summarize any text, URL or PDF + ask questions with RAG</div></div>", unsafe_allow_html=True)
    input_mode = st.radio("Input type", options=["Paste text", "Provide URL", "Upload PDF"], index=0)

    user_text = ""
    uploaded_file = None
    url = ""
    if input_mode == "Paste text":
        user_text = st.text_area("Paste your document / transcript here", height=250)
    elif input_mode == "Provide URL":
        url = st.text_input("Enter a webpage URL (http(s)...)")
        if url:
            st.markdown("Fetched text preview (first 5 KB):")
            fetched = fetch_text_from_url(url)
            st.code(fetched[:5000] + ("..." if len(fetched)>5000 else ""))
            user_text = fetched
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("Reading PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
            st.markdown("PDF text preview (first 5 KB):")
            st.code(pdf_text[:5000] + ("..." if len(pdf_text)>5000 else ""))
            user_text = pdf_text

    # Summarization controls
    st.markdown("---")
    st.subheader("Summarization settings")
    chunk_size = st.slider("Chunk size (words)", 500, 3000, 1000, 100)
    overlap = st.slider("Chunk overlap (words)", 0, 1200, 200, 50)
    top_k = st.slider("Retrieval top K (for RAG context)", 1, 10, 4)
    summarize_button = st.button("Summarize")

with col2:
    st.markdown("### Quick tips")
    st.write("""
    - For long documents, upload the PDF or paste long text.
    - URL extraction uses simple heuristics; for paywalled or JS-heavy sites it may fail.
    - Set `OPENAI_API_KEY` as an environment variable before launching.
    - Costs: embedding + chat calls use your OpenAI quota.
    """)
    st.markdown("---")
    st.markdown("### Recent Actions")
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.write("\n".join(st.session_state.history[-6:]))

# Main processing
if summarize_button:
    if not user_text or len(user_text.strip()) < 20:
        st.warning("Please provide at least some text (paste text, fetch a URL, or upload a PDF).")
    else:
        with st.spinner("Chunking text..."):
            chunks = chunk_text(user_text, chunk_size=chunk_size, overlap=overlap)
        st.success(f"Document split into {len(chunks)} chunks.")
        # embed chunks
        with st.spinner("Computing embeddings... (this may cost API credits)"):
            vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)
        st.session_state.history.append(f"Indexed {len(chunks)} chunks at {time.strftime('%H:%M:%S')}")

        # Build retrieval context (top_k)
        # Use a short generic query to capture central themes: use the document title heuristic if URL or PDF filename available
        seed_query = "Summarize the document main ideas"
        retrieved = retrieve_top_k(seed_query, chunks, index, vectors, top_k=top_k)
        retrieved_texts = [r[2] for r in retrieved]

        with st.spinner("Generating summary..."):
            abstract, points = generate_summary_and_points(user_text, retriever_context=retrieved_texts)

        # Display results
        st.markdown("## Summary Abstract")
        st.write(abstract)
        st.markdown("## Key Points")
        st.write(points or "No key points extracted.")
        # show retrieved sources
        st.markdown("### Top retrieved chunks used for context")
        for idx, score, txt in retrieved:
            st.markdown(f"<div class='source-box'><strong>Chunk {idx}</strong> (score={score:.3f})<br/><small>{txt[:800]}{'...' if len(txt)>800 else ''}</small></div>", unsafe_allow_html=True)

        # Save state for Q&A
        st.session_state['rag_context'] = {
            "chunks": chunks,
            "vectors": vectors,
            "index": index
        }
        st.success("RAG index and context ready for Q&A.")

# Q&A box
st.markdown("---")
st.header("Ask a question about the document (RAG)")
q_col1, q_col2 = st.columns([3,1])
with q_col1:
    question = st.text_input("Type a question about the current document")
with q_col2:
    ask_button = st.button("Ask")

if ask_button:
    if 'rag_context' not in st.session_state:
        st.warning("You must first Summarize (build the RAG index) before asking questions.")
    elif not question.strip():
        st.warning("Type a question first.")
    else:
        rag = st.session_state.rag_context
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve_top_k(question, rag["chunks"], rag["index"], rag["vectors"], top_k=5)
        st.markdown("**Retrieved context snippets:**")
        for idx, score, txt in retrieved:
            st.markdown(f"- [CHUNK {idx}] (score={score:.3f}) {txt[:300]}{'...' if len(txt)>300 else ''}")
        with st.spinner("Answering using retrieved context..."):
            answer = answer_question_with_context(question, retrieved)
        st.markdown("### Answer")
        st.write(answer)
        st.session_state.history.append(f"Q: {question}  — answered at {time.strftime('%H:%M:%S')}")

# Footer: allow download of summary
st.markdown("---")
if st.button("Download last summary"):
    # create a simple text file with last summary from session if present
    if 'rag_context' in st.session_state:
        # we reconstruct using displayed variables in session if they exist
        summary_text = ""
        # attempt to fetch last displayed abstract/points from the page by regenerating (cheap) or storing earlier
        summary_text = "Use the Summarize button to generate a summary and then click download."
        st.download_button("Download (TXT)", summary_text, file_name="summary.txt", mime="text/plain")
    else:
        st.warning("No summary available. Run Summarize first.")
