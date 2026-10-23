import streamlit as st
import pandas as pd
from openai import OpenAI, RateLimitError
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dynamic AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Dynamic AI Chatbot")

# Custom Clean Styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Initialize Groq Client via Streamlit Secrets
# ---------------------------------------------------------
api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found. Configure it in .streamlit/secrets.toml locally or in Streamlit Cloud Secrets.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("⚙️ Chatbot Settings")

persona_options = {
    "General Assistant": "You are a helpful, dynamic, and versatile AI assistant.",
    "Data Science Tutor": "You are an expert Data Science tutor. Explain concepts clearly with concise Python snippets.",
    "SQL Specialist": "You are an expert SQL Developer and Database Architect. Assist with complex queries, indexing, and schema design.",
    "ML Engineer": "You are a Senior Machine Learning Engineer. Help with model building, evaluation metrics, feature engineering, and pipeline design.",
    "Code Reviewer": "You are a Senior Software Engineer. Analyze code for bugs, efficiency, and clean structure.",
    "Resume & Interview Coach": "You are an expert HR and Career Coach. Help refine technical resume bullet points and provide mock interview preparation.",
    "Creative Writer": "You are an imaginative creative writer providing engaging and polished prose.",
    "Custom Persona": "Custom"
}

selected_persona = st.sidebar.selectbox("Choose AI Persona", list(persona_options.keys()))

if selected_persona == "Custom Persona":
    system_prompt = st.sidebar.text_area(
        "Enter Custom System Instructions:",
        value="You are a helpful specialized assistant."
    )
else:
    system_prompt = persona_options[selected_persona]

temperature = st.sidebar.slider("Creativity (Temperature)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# File Context Analysis
st.sidebar.markdown("---")
st.sidebar.subheader("📄 File Context Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Text File", type=["csv", "txt"], key="file_context_uploader")

file_context = ""
auto_file_prompt = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            file_context = f"\n\n[USER ATTACHED DATASET SUMMARY]:\nFilename: {uploaded_file.name}\nColumns: {list(df.columns)}\nFirst 3 Rows:\n{df.head(3).to_string()}"
            st.sidebar.success(f"Loaded CSV: {uploaded_file.name}")
        elif uploaded_file.name.endswith(".txt"):
            text_content = uploaded_file.read().decode("utf-8")
            file_context = f"\n\n[USER ATTACHED FILE CONTENT]:\n{text_content[:1500]}"
            st.sidebar.success(f"Loaded Text: {uploaded_file.name}")
            
        # Trigger automatic prompt on initial upload
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != uploaded_file.name:
            st.session_state["last_uploaded_file"] = uploaded_file.name
            auto_file_prompt = f"I uploaded `{uploaded_file.name}`. Please analyze this file and summarize its structure and contents."
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
elif "last_uploaded_file" in st.session_state:
    del st.session_state["last_uploaded_file"]

# Session State Setup
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_prompt + file_context}]
else:
    st.session_state["messages"][0] = {"role": "system", "content": system_prompt + file_context}

# Sidebar Session Controls
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Controls")

user_msg_count = sum(1 for m in st.session_state["messages"] if m["role"] != "system" and m["role"] == "user")
st.sidebar.metric("Messages Sent", user_msg_count)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [{"role": "system", "content": system_prompt + file_context}]
    if "last_uploaded_file" in st.session_state:
        del st.session_state["last_uploaded_file"]
    st.toast("Chat history cleared!", icon="🧹")
    st.rerun()

chat_text = "\n\n".join([
    f"[{m.get('time', 'N/A')}] {m['role'].upper()}: {m['content']}" 
    for m in st.session_state["messages"] if m["role"] != "system"
])

st.sidebar.download_button(
    label="📥 Download Chat (.txt)",
    data=chat_text,
    file_name="chat_history.txt",
    mime="text/plain"
)

# Active Header Display
st.caption(f"Active Persona: **{selected_persona}**")
st.markdown("---")

# Render Chat History
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "time" in msg:
                st.caption(f"🕒 {msg['time']}")

# ---------------------------------------------------------
# Front-Side Voice Note Popover & Text Input
# ---------------------------------------------------------
user_prompt = None

with st.popover("🎙️ Voice Note", use_container_width=False):
    audio_val = st.audio_input("Record audio note", key="front_audio_recorder")
    if audio_val:
        with st.spinner("Transcribing..."):
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=("audio.wav", audio_val.read(), "audio/wav")
                )
                user_prompt = transcription.text
                st.success(f"Transcribed: \"{user_prompt}\"")
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")

text_input = st.chat_input("Message Dynamic AI Chatbot...")

# Determine effective prompt (voice note, typed text, or automatic file upload trigger)
if text_input:
    user_prompt = text_input
elif auto_file_prompt:
    user_prompt = auto_file_prompt

# ---------------------------------------------------------
# Handle Execution & Streaming Response
# ---------------------------------------------------------
if user_prompt:
    current_time = datetime.now().strftime("%I:%M %p")
    
    st.session_state["messages"].append({
        "role": "user", 
        "content": user_prompt,
        "time": current_time
    })
    
    with st.chat_message("user"):
        st.markdown(user_prompt)
        st.caption(f"🕒 {current_time}")

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]],
                temperature=temperature,
                stream=True
            )
            response_text = st.write_stream(stream)
            reply_time = datetime.now().strftime("%I:%M %p")
            st.caption(f"🕒 {reply_time}")

            st.session_state["messages"].append({
                "role": "assistant", 
                "content": response_text,
                "time": reply_time
            })
        except RateLimitError:
            st.error("⏳ Rate limit reached on Groq free tier. Click 'Clear Chat History' in the sidebar.")
            st.session_state["messages"].pop()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state["messages"].pop()