import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from helpers.rag import perform_rag, fetch_repos_from_pincone
from helpers.repo import process_repo

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Initialize session state variables
if "selected_codebase" not in st.session_state:
    st.session_state.selected_codebase = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repos_fetched" not in st.session_state:
    st.session_state.repos_fetched = False

if "repos" not in st.session_state:
    st.session_state.repos = []

def fetch_codebase_options():
    if not st.session_state.repos_fetched:
        st.session_state.repos = fetch_repos_from_pincone()
        st.session_state.repos_fetched = True

def select_codebase():
    fetch_codebase_options()

    # Select box for indexed codebases
    selected = st.selectbox(
        "Select a codebase already indexed to Pinecone:",
        options=st.session_state.repos,
        index=st.session_state.repos.index(st.session_state.selected_codebase)
        if st.session_state.selected_codebase in st.session_state.repos
        else 0
    )

    github_repo_url = st.text_input("Or paste a public GitHub Link:", placeholder="https://github.com/username/repo")
    process_github = st.button("Index GitHub Repository")

    # Process the GitHub repository if provided
    if process_github and github_repo_url:
        with st.spinner('Uploading Repository Data to Pinecone...'):
            process_repo(github_repo_url)
        st.success(f"{github_repo_url} uploaded to Pinecone successfully!")
        # Refetch repos to include the newly indexed one
        st.session_state.repos_fetched = False
        st.session_state.selected_codebase = None

    # Update the selected codebase when user selects a new one
    if selected != st.session_state.selected_codebase:
        st.session_state.selected_codebase = selected

def main():
    st.title("💬 Codebase RAG Chat")

    # Display the selected codebase if available
    if st.session_state.selected_codebase:
        st.write(f"Codebase in use: {st.session_state.selected_codebase}")
    else:
        st.warning("No codebase selected. Please select or index a new one.")

    # Chat section: display existing messages
    if st.session_state.selected_codebase:
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Handle new chat input
        prompt = st.chat_input("Chat...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Generate assistant response
                llm_response = perform_rag(client, prompt, st.session_state.selected_codebase)
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
                with st.chat_message("assistant"):
                    st.markdown(llm_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar for codebase selection
with st.sidebar:
    st.header("Codebase Selection")
    select_codebase()

# Run the main function
if __name__ == "__main__":
    main()
