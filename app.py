import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Dynamic AI Chatbot", page_icon="🤖")
st.title("🤖 Dynamic AI Chatbot")

# Initialize Groq Client
if "GROQ_API_KEY" in st.secrets:
  client = OpenAI(
      base_url="https://api.groq.com/openai/v1",
      api_key=st.secrets["GROQ_API_KEY"],
  )
else:
  st.error(
      "❌ GROQ_API_KEY is missing! Please add it in Streamlit Cloud Settings ->"
      " Secrets."
  )
  client = None

# Sidebar Configuration
st.sidebar.header("Configuration")
selected_persona = st.sidebar.selectbox(
    "Choose AI Persona",
    ["General Assistant", "Data Analyst", "Creative Writer", "Python Expert"],
)

uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type=["csv"])
file_summary = ""
auto_trigger_prompt = ""

if uploaded_file is not None:
  try:
    df = pd.read_csv(uploaded_file)
    file_summary = f"\n\n[Uploaded CSV Data Preview & Columns]:\nColumns: {list(df.columns)}\n{df.head(5).to_string()}"
    st.sidebar.success("File loaded successfully!")

    # Automatically set a trigger prompt if a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
      st.session_state.last_uploaded = uploaded_file.name
      auto_trigger_prompt = (
          "I have uploaded a CSV file named"
          f" {uploaded_file.name}. Please analyze the columns, summarize the"
          " dataset, and tell me what insights you can find from it."
      )
  except Exception as e:
    st.sidebar.error(f"Error reading file: {e}")

# Chat History
if "messages" not in st.session_state:
  st.session_state.messages = []

# If a file was just uploaded, automatically add the prompt to chat history
if auto_trigger_prompt:
  st.session_state.messages.append(
      {"role": "user", "content": auto_trigger_prompt}
  )

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Handle auto-trigger response if file was just uploaded
if auto_trigger_prompt and client:
  with st.chat_message("assistant"):
    with st.spinner("Analyzing your CSV file..."):
      try:
        system_prompts = {
            "General Assistant": (
                "You are a helpful, accurate, and polite AI assistant."
            ),
            "Data Analyst": (
                "You are an expert data analyst specializing in Python, Pandas,"
                " SQL, and data visualization. Analyze the provided dataset"
                " thoroughly."
            ),
            "Creative Writer": (
                "You are a creative writer skilled in storytelling and copy."
            ),
            "Python Expert": (
                "You are an expert Python developer who writes clean code."
            ),
        }
        base_system_prompt = system_prompts.get(
            selected_persona, system_prompts["General Assistant"]
        )
        current_system_content = base_system_prompt + file_summary

        messages_payload = [
            {"role": "system", "content": current_system_content}
        ]
        for m in st.session_state.messages[:-1]:
          messages_payload.append({"role": m["role"], "content": m["content"]})
        messages_payload.append(
            {"role": "user", "content": auto_trigger_prompt}
        )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages_payload,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
      except Exception as e:
        error_message = f"⚠️ API Error: {e}"
        st.error(error_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )

# Standard User Input Box
if prompt := st.chat_input("Type your message here..."):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  if client:
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          system_prompts = {
              "General Assistant": (
                  "You are a helpful, accurate, and polite AI assistant."
              ),
              "Data Analyst": (
                  "You are an expert data analyst specializing in Python, Pandas,"
                  " SQL, and data visualization. Analyze the provided dataset"
                  " thoroughly."
              ),
              "Creative Writer": (
                  "You are a creative writer skilled in storytelling and copy."
              ),
              "Python Expert": (
                  "You are an expert Python developer who writes clean code."
              ),
          }
          base_system_prompt = system_prompts.get(
              selected_persona, system_prompts["General Assistant"]
          )
          current_system_content = base_system_prompt + file_summary

          messages_payload = [
              {"role": "system", "content": current_system_content}
          ]
          for m in st.session_state.messages:
            messages_payload.append({"role": m["role"], "content": m["content"]})

          completion = client.chat.completions.create(
              model="openai/gpt-oss-20b",
              messages=messages_payload,
              temperature=0.7,
          )
          answer = completion.choices[0].message.content
          st.markdown(answer)
          st.session_state.messages.append(
              {"role": "assistant", "content": answer}
          )
        except Exception as e:
          error_message = f"⚠️ API Error: {e}"
          st.error(error_message)
          st.session_state.messages.append(
              {"role": "assistant", "content": error_message}
          )
  else:
    st.error("Client not initialized. Check your Groq API key secrets.")
