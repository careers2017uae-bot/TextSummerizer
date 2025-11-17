import streamlit as st
import os
import requests
from groq import Groq
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------------
# Groq Client
# -------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------
# PDF Extraction
# -------------------------------------
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        txt = page.extract_text()
        if txt:
            text += txt + "\n"
    return text

# -------------------------------------
# URL Extraction
# -------------------------------------
def extract_from_url(url):
    try:
        response = requests.get(url, timeout=15)
        return response.text if response.status_code == 200 else "Error fetching URL."
    except:
        return "Invalid or unreachable URL."

# -------------------------------------
# Custom Text Splitter
# -------------------------------------
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# -------------------------------------
# Embeddings + FAISS DB
# -------------------------------------
def build_faiss_db(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings, model

# -------------------------------------
# Retrieve Similar Chunks
# -------------------------------------
def retrieve_context(query, index, embeddings, model, chunks, k=5):
    q_emb = model.encode([query])
    distances, idx = index.search(q_emb, k)
    return "\n\n".join(chunks[i] for i in idx[0])

# -------------------------------------
# Groq LLM Call
# -------------------------------------
def groq_chat(prompt):
    try:
        # Safety: truncate prompt to 120,000 chars
        if len(prompt) > 120_000:
            prompt = prompt[:120_000]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # UPDATED MODEL
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"


# -------------------------------------
# Streamlit UI
# -------------------------------------
st.set_page_config(page_title="GROQ Text Summarizer + RAG", layout="wide")
st.write("Groq Key Loaded?", os.getenv("GROQ_API_KEY") is not None)

st.title("📘 GROQ-Based Text Summarizer + RAG QA")
st.write("Upload a document, summarize it, and ask questions using RAG.")

input_mode = st.radio("Select Input Mode:", ["Text", "PDF", "URL"])

document_text = ""

if input_mode == "Text":
    document_text = st.text_area("Enter Text:", height=250)

elif input_mode == "PDF":
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf:
        document_text = extract_text_from_pdf(pdf)
        st.success("PDF extracted successfully!")

elif input_mode == "URL":
    url = st.text_input("Enter URL:")
    if url:
        document_text = extract_from_url(url)
        st.success("URL content fetched!")

# -----------------------
# Summarization
# -----------------------
if st.button("Summarize Document"):
    if not document_text.strip():
        st.error("Please provide content first.")
    else:
        with st.spinner("Summarizing with Groq…"):

            short_doc = document_text[:40_000]  # avoid over-limit input

            prompt = f"""
Summarize the following document into a clear, concise paragraph 
with the main ideas only.

DOCUMENT:
{short_doc}
"""

            summary = groq_chat(prompt)

        st.subheader("📌 Summary")
        st.write(summary)


        # Build FAISS DB for RAG
        with st.spinner("Building RAG vector database…"):
            chunks = split_text(document_text)
            index, embeddings, model = build_faiss_db(chunks)

        st.success("RAG Ready! Ask questions below.")

        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("Retrieving answer with RAG…"):
                context = retrieve_context(query, index, embeddings, model, chunks)

                final_prompt = f"""
Answer the question using ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, short answer.
"""
                answer = groq_chat(final_prompt)

            st.subheader("🧠 RAG Answer")
            st.write(answer)
