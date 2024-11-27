import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from helpers.rag import perform_rag
from helpers.repo import process_repo

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Show title and description.
st.title("💬 Codebase RAG")
st.write(
    "This is a chatbot that uses Groq's Llama 3.1 Versatile model to generate responses."
)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Assign Default repo url
codebase = "https://github.com/desantamaria/discord-clone"

github_repo_url = st.text_input("Enter a GitHub repository url")

if github_repo_url:
    with st.spinner('Uploading Repository Data to Pinecone...'):
        process_repo(github_repo_url)
    st.write(
    f"{github_repo_url} uploaded to Pinecone successfully!"
    )
    codebase = github_repo_url

st.write(
    f"Codebase in use: {codebase}"
)

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Chat..."):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the Groq API.    
    llm_response = perform_rag(client, prompt, codebase)

    # Write the response to the chat using `st.write`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write(llm_response)
    st.session_state.messages.append({"role": "assistant", "content": response})
