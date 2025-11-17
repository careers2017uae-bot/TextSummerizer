import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found! Please set it in your environment.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI Setup
st.set_page_config(page_title="Groq Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 Groq AI — Smart Text Summarizer")
st.write("Paste text or upload a document. Summaries are powered by **LLaMA 3.1** on Groq.")

# Input
user_text = st.text_area("Enter text to summarize:", height=250)

summary_length = st.selectbox(
    "Select summary style",
    ["Short", "Medium", "Detailed"],
    index=1
)

# Mapping summary styles
length_map = {
    "Short": "Summarize the text in 3–4 crisp bullet points.",
    "Medium": "Provide a single concise paragraph summary.",
    "Detailed": "Provide a detailed summary of 2–3 paragraphs with key details."
}

def summarize(text, style_instruction):
    """Send summarization request to Groq."""
    prompt = f"""
You are an expert summarizer. {style_instruction}

Summarize the following text:
\"\"\"{text}\"\"\"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # NEW SAFE MODEL
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )

    return response.choices[0].message.content


# Button
if st.button("Summarize"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text before summarizing.")
        st.stop()

    with st.spinner("Generating summary..."):
        try:
            summary = summarize(user_text, length_map[summary_length])
            st.subheader("📌 Summary")
            st.success(summary)
        except Exception as e:
            st.error(f"⚠️ Groq Error: {e}")
