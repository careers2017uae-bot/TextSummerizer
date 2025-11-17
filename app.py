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

# App UI
st.set_page_config(page_title="Groq Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 Text Summarizer using Groq LLM")
st.write("Paste any text below and get a clean, concise summary powered by Groq models.")

user_text = st.text_area("Enter text to summarize", height=250)

summary_length = st.selectbox(
    "Summary Length",
    ["Short", "Medium", "Detailed"],
    index=0
)

if st.button("Summarize"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text before summarizing.")
        st.stop()

    # Summary instruction based on user selection
    length_map = {
        "Short": "Summarize the text in 3–4 bullet points.",
        "Medium": "Provide a short paragraph summary.",
        "Detailed": "Summarize with maximum clarity in 2–3 detailed paragraphs."
    }

    prompt = f"""
You are an expert summarizer. {length_map[summary_length]}

Text to summarize:
\"\"\"{user_text}\"\"\"
"""

    try:
        # Groq Chat Completion
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.4
        )

        summary = response.choices[0].message.content

        st.subheader("📌 Summary")
        st.success(summary)

    except Exception as e:
        st.error(f"⚠️ Groq Error: {str(e)}")
