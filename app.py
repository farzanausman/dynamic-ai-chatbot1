from datetime import datetime
import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize Streamlit Page Config
st.set_page_config(
    page_title="Dynamic AI Chatbot", page_icon="🤖", layout="centered"
)

st.title("🤖 Dynamic AI Chatbot")

# Initialize Groq Client using secrets
if "GROQ_API_KEY" in st.secrets:
  client = OpenAI(
      base_url="https://api.groq.com/openai/v1",
      api_key=st.secrets["GROQ_API_KEY"],
  )
else:
  st.error(
      "GROQ_API_KEY not found. Configure it in .streamlit/secrets.toml locally"
      " or in Streamlit Cloud Secrets."
  )
  client = None

# Sidebar Configuration (Temperature Slider is removed)
st.sidebar.header("Configuration")
selected_persona = st.sidebar.selectbox(
    "Choose AI Persona",
    [
        "General Assistant",
        "Data Analyst",
        "Creative Writer",
        "Python Expert",
    ],
)

# Optional file uploader for context analysis
uploaded_file = st.sidebar.file_uploader(
    "Upload Document/CSV (Optional)", type=["csv", "txt"]
)
file_context = ""

if uploaded_file is not None:
  if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
    file_context = (
        f"\n\nHere is uploaded CSV data summary:\n{df.head().to_string()}"
    )
    st.sidebar.success("CSV uploaded successfully!")
  else:
    file_context = f"\n\nUploaded file content:\n{uploaded_file.read().decode('utf-8')}"
    st.sidebar.success("Text file uploaded successfully!")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display Prior Chat Messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Type your message here..."):
  # Append user message to history
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Generate Assistant Response
  if client:
    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      full_response = ""

      # Define Persona System Prompts
      system_prompts = {
          "General Assistant": (
              "You are a helpful, accurate, and polite AI assistant."
          ),
          "Data Analyst": (
              "You are an expert data analyst specializing in Python, Pandas,"
              " SQL, and data visualization."
          ),
          "Creative Writer": (
              "You are a creative writer skilled in storytelling, poetry, and"
              " engaging copy."
          ),
          "Python Expert": (
              "You are an expert Python developer who writes clean, efficient,"
              " and well-documented code."
          ),
      }

      current_system_prompt = (
          system_prompts.get(selected_persona, system_prompts["General Assistant"])
          + file_context
      )

      # Build messages payload for API
      messages_payload = [{"role": "system", "content": current_system_prompt}]
      for m in st.session_state.messages:
        messages_payload.append({"role": m["role"], "content": m["content"]})

      try:
        # API call with fixed temperature (slider removed)
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages_payload,
            temperature=0.7,  # Fixed default temperature value
        )
        full_response = completion.choices[0].message.content
        message_placeholder.markdown(full_response)
      except Exception as e:
        full_response = f"An error occurred: {e}"
        message_placeholder.error(full_response)

    # Save assistant response to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
