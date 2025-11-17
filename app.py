import streamlit as st
import requests
import os
import tempfile
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

# -------------------------------
# Initialize GROQ Client
# -------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Helper: Extract text from PDF
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------------------
# Helper: Extract text from URL
# -------------------------------
def extract_from_url(url):
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            return response.text
        return "Error fetching URL content."
    except:
        return "Invalid or unreachable URL."

# -------------------------------
# LLM CALL (Groq)
# -------------------------------
def groq_chat(prompt, model="mixtral-8x7b-32768"):
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800
    )
    return chat_completion.choices[0].message["content"]

# -------------------------------
# RAG: Build FAISS Vector DB
# -------------------------------
def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = FAISS.from_texts(chunks, embedder)

    return vectordb

# -------------------------------
# RAG QUERY
# -------------------------------
def rag_query(query, vectordb):
    docs = vectordb.similarity_search(query, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an expert assistant. A user asked a question about a document.
Use ONLY the following retrieved context to answer:

CONTEXT:
{context}

QUESTION:
{query}

Provide a clear, correct, concise answer.
"""
    return groq_chat(prompt)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Groq Text Summarizer + RAG QA", layout="wide")

st.title("📘 GROQ-Powered Text Summarizer + RAG QA")
st.write("Upload text, PDF, or a URL → Summarize → Ask questions about it using RAG.")

# Input mode selection
input_mode = st.radio("Choose input method:", ["Text", "PDF", "URL"])

document_text = ""

if input_mode == "Text":
    document_text = st.text_area("Paste your document text here:", height=250)

elif input_mode == "PDF":
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF…"):
            document_text = extract_text_from_pdf(uploaded_pdf)
            st.success("PDF text extracted successfully!")

elif input_mode == "URL":
    url = st.text_input("Enter URL:")
    if url:
        with st.spinner("Fetching web page…"):
            document_text = extract_from_url(url)
            st.success("URL content extracted!")

# Summarize Button
if st.button("Summarize Document"):
    if not document_text.strip():
        st.error("Please provide text, PDF, or URL content first.")
    else:
        with st.spinner("Summarizing with GROQ…"):
            summary_prompt = f"""
Summarize the following document into a clear, concise paragraph.
Focus only on main ideas. Avoid unnecessary details.

DOCUMENT:
{document_text}
"""
            summary = groq_chat(summary_prompt)

        st.subheader("📌 Summary")
        st.write(summary)

        # Build Vector DB for RAG
        with st.spinner("Building vector DB for RAG…"):
            vectordb = build_vector_db(document_text)

        st.success("RAG Ready! Ask questions below.")

        # RAG Q/A
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            with st.spinner("Searching document + answering using RAG…"):
                answer = rag_query(user_question, vectordb)
            st.subheader("🧠 RAG Answer")
            st.write(answer)
