# app.py (Updated for GROQ API)

import os
import io
import time
from typing import List, Tuple

import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import numpy as np
import faiss
from groq import Groq

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it before running the app.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# GROQ MODELS (Updated — no decommissioned models)
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama-3.1-70b-versatile"

# Embedding dimension for nomic-embed-text
EMBED_DIM = 768

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

# ---------- GROQ Embeddings + FAISS ----------
def embed_texts(texts: List[str]) -> List[np.ndarray]:
    vectors = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )

        for item in resp.data:
            vectors.append(np.array(item.embedding, dtype=np.float32))

    return vectors

def build_faiss_index(vectors: List[np.ndarray]):
    if not vectors:
        return None

    vecs = np.vstack(vectors).astype(np.float32)

    # normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-12)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index

def retrieve_top_k(query, chunks, index, vectors, top_k=5):
    q_vec = embed_texts([query])[0].astype(np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    q = q_vec.reshape(1, -1)

    D, I = index.search(q, top_k)

    results = [(int(idx), float(score), chunks[idx]) for score, idx in zip(D[0], I[0]) if 0 <= idx < len(chunks)]
    return results

# ---------- GROQ Chat ----------
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

def generate_summary_and_points(full_text: str, retriever_context=None):
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

# ---------- UI ----------
st.set_page_config(page_title="Groq RAG Summarizer", page_icon="⚡", layout="wide")

st.title("⚡ Groq-Powered RAG Text Summarizer")

input_mode = st.radio("Input Type", ["Paste text", "URL", "PDF"])

if input_mode == "Paste text":
    user_text = st.text_area("Paste your text", height=250, key="txt_input")

elif input_mode == "URL":
    url = st.text_input("Enter URL")
    if url:
        fetched = fetch_text_from_url(url)
        st.code(fetched[:4000])
        user_text = fetched

else:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        pdf_text = extract_text_from_pdf(uploaded)
        st.code(pdf_text[:4000])
        user_text = pdf_text

chunk_size = st.slider("Chunk size (words)", 500, 3000, 1000)
overlap = st.slider("Chunk overlap", 0, 1000, 200)
top_k = st.slider("Top-K Retrieval", 1, 10, 4)

if st.button("Summarize"):
    if not user_text or len(user_text.strip()) < 20:
        st.warning("Please enter valid text.")
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
