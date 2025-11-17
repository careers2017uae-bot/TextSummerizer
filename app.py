# app.py
import os
import time
from typing import List, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it before running the app.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
CHAT_MODEL = "llama-3.3-70b-versatile"

# Local embeddings model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384  # MiniLM embedding dimension

# ---------- Utilities ----------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.warning(f"PDF parsing failed: {e}")
        return ""

def fetch_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "GroqRAG/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        paragraphs = [p for p in paragraphs if len(p) > 60]
        return "\n\n".join(paragraphs) or soup.get_text(separator=" ", strip=True)
    except Exception as e:
        st.warning(f"Unable to fetch URL: {e}")
        return ""

def chunk_text(text: str, chunk_size:int = 1000, overlap:int = 200) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

# ---------- Embeddings + FAISS ----------
def embed_texts(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(vectors: np.ndarray):
    if vectors is None or len(vectors) == 0:
        return None
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def retrieve_top_k(query:str, chunks:List[str], index:faiss.IndexFlatIP, vectors:np.ndarray, top_k:int=5):
    q_vec = embed_texts([query])[0].astype(np.float32).reshape(1, -1)
    D, I = index.search(q_vec, top_k)
    results = [(int(idx), float(score), chunks[idx]) for score, idx in zip(D[0], I[0]) if 0 <= idx < len(chunks)]
    return results

# ---------- Groq Chat ----------
def chat_completion(system_prompt, user_prompt, max_tokens=500):
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq Error: {e}"

def generate_summary_and_points(full_text: str, retriever_context=None) -> Tuple[str,str]:
    context = "\n\n---\n\n".join(retriever_context) if retriever_context else ""
    system_prompt = (
        "You are a concise summarizer. "
        "Return:\n1) 2–3 sentence abstract\n2) 5 bullet key points"
    )
    user_prompt = f"{context}\n\n{full_text}"
    out = chat_completion(system_prompt, user_prompt)
    if "\n1" in out:
        abstract, points = out.split("\n1", 1)
        return abstract.strip(), "1" + points.strip()
    return out, ""

def answer_question_with_context(question: str, retrieved_chunks: List[Tuple[int,float,str]]) -> str:
    context_text = "\n\n".join([f"### CHUNK {idx} (score={score:.3f})\n{chunk}" 
                                for idx, score, chunk in retrieved_chunks])
    system_prompt = (
        "You are a helpful assistant that answers questions using only the provided context. "
        "If the answer is not in the context, say 'I don't know based on the provided document.'"
    )
    user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
    return chat_completion(system_prompt, user_prompt, max_tokens=400)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Groq RAG Summarizer", page_icon="✂️", layout="wide")
st.title("⚡ GROQ + FAISS RAG Text Summarizer and Q/A by Engr Bilal")

input_mode = st.radio("Input Type", ["Paste text", "URL", "PDF"])
user_text = ""

if input_mode == "Paste text":
    user_text = st.text_area("Paste your document here", height=250, key="txt_input")
elif input_mode == "URL":
    url = st.text_input("Enter webpage URL")
    if url:
        user_text = fetch_text_from_url(url)
        st.code(user_text[:4000])
else:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        user_text = extract_text_from_pdf(uploaded)
        st.code(user_text[:4000])

# Summarization settings
chunk_size = st.slider("Chunk size (words)", 500, 3000, 1000)
overlap = st.slider("Chunk overlap", 0, 1000, 200)
top_k = st.slider("Top-K Retrieval", 1, 10, 4)

if st.button("Summarize"):
    if not user_text.strip():
        st.warning("Please provide valid text input.")
    else:
        chunks = chunk_text(user_text, chunk_size, overlap)
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)
        retrieved = retrieve_top_k("Main ideas", chunks, index, vectors, top_k)
        retrieved_texts = [r[2] for r in retrieved]
        abstract, points = generate_summary_and_points(user_text, retrieved_texts)

        st.subheader("📌 Abstract")
        st.write(abstract)

        st.subheader("📌 Key Points")
        st.write(points)

        # Save for Q&A
        st.session_state['rag_context'] = {"chunks": chunks, "vectors": vectors, "index": index}

# Q&A
st.markdown("---")
st.header("Ask a question about the document")
question = st.text_input("Type your question")
if st.button("Ask Question"):
    if 'rag_context' not in st.session_state:
        st.warning("You must first Summarize the document before asking questions.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        rag = st.session_state['rag_context']
        retrieved = retrieve_top_k(question, rag["chunks"], rag["index"], rag["vectors"], top_k=5)
        st.markdown("**Retrieved snippets:**")
        for idx, score, txt in retrieved:
            st.markdown(f"- [CHUNK {idx}] (score={score:.3f}) {txt[:300]}{'...' if len(txt)>300 else ''}")
        answer = answer_question_with_context(question, retrieved)
        st.markdown("### Answer")
        st.write(answer)
