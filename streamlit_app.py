import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from helpers.rag import perform_rag, fetch_repos_from_pincone
from helpers.repo import process_repo
import google.generativeai as genai


# Load environment variables
load_dotenv()

# Initialize API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize clients for different AI models
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if "selected_codebase" not in st.session_state:
    st.session_state.selected_codebase = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repos_fetched" not in st.session_state:
    st.session_state.repos_fetched = False

if "repos" not in st.session_state:
    st.session_state.repos = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Groq's Llama 3.1"  # Default model


# Fetch available codebases from Pinecone
def fetch_codebase_options():
    if not st.session_state.repos_fetched:
        st.session_state.repos = fetch_repos_from_pincone()
        st.session_state.repos_fetched = True

# Function to handle codebase selection
def select_codebase():
    fetch_codebase_options()

    selected = st.selectbox(
        "Select a codebase already indexed to Pinecone:",
        options=st.session_state.repos,
        index=st.session_state.repos.index(st.session_state.selected_codebase)
        if st.session_state.selected_codebase in st.session_state.repos
        else 0
    )

    github_repo_url = st.text_input("Or paste a public GitHub Link:", placeholder="https://github.com/username/repo")
    process_github = st.button("Index GitHub Repository")

    if process_github and github_repo_url:
        with st.spinner('Uploading Repository Data to Pinecone...'):
            process_repo(github_repo_url)
        st.success(f"{github_repo_url} uploaded to Pinecone successfully!")
        st.session_state.repos_fetched = False
        st.session_state.selected_codebase = None

    if selected != st.session_state.selected_codebase:
        st.session_state.selected_codebase = selected

# Function to select AI model
def select_model():
    model_options = ["Groq's Llama 3.1", "Google Gemini"]
    selected_model = st.selectbox(
        "Select AI model:",
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

# Main function for chat
def main():
    st.title("💬 Codebase RAG Chat")

    if st.session_state.selected_codebase:
        st.write(f"Codebase in use: {st.session_state.selected_codebase}")
    else:
        st.warning("No codebase selected. Please select or index a new one.")

    # Chat section
    if st.session_state.selected_codebase:
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("Chat...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Pass selected model to perform_rag function
                llm_response = perform_rag(
                    groq_client if st.session_state.selected_model == "Groq's Llama 3.1" else genai, 
                    prompt, 
                    st.session_state.selected_codebase, 
                    st.session_state.selected_model
                )

                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                with st.chat_message("assistant"):
                    st.markdown(llm_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar for codebase and model selection
with st.sidebar:
    st.header("Codebase Selection")
    select_codebase()
    st.header("AI Model Selection")
    select_model()

# Run the main function
if __name__ == "__main__":
    main()
