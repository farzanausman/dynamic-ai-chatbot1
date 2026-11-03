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

# Sidebar Configuration
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
    try:
      df = pd.read_csv(uploaded_file)
      # Provide column names and preview data explicitly to the model
      file_context = (
          f"\n\n[Context from uploaded file: {uploaded_file.name}]\n"
          f"Columns: {list(df.columns)}\n"
          f"Data Preview (First 10 rows):\n{df.head(10).to_string()}"
      )
      st.sidebar.success("CSV uploaded successfully!")
    except Exception as e:
      st.sidebar.error(f"Error reading CSV: {e}")
  else:
    try:
      file_context = f"\n\n[Context from uploaded file: {uploaded_file.name}]\n{uploaded_file.read().decode('utf-8')}"
      st.sidebar.success("Text file uploaded successfully!")
    except Exception as e:
      st.sidebar.error(f"Error reading file: {e}")

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
              " SQL, and data visualization. Analyze the provided dataset"
              " accurately."
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

      base_system_prompt = system_prompts.get(
          selected_persona, system_prompts["General Assistant"]
      )

      # Build messages payload for API, injecting file context into system prompt
      current_system_content = base_system_prompt + file_context
      messages_payload = [{"role": "system", "content": current_system_content}]

      for m in st.session_state.messages:
        messages_payload.append({"role": m["role"], "content": m["content"]})

      try:
        # API call with fixed temperature
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages_payload,
            temperature=0.7,
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
