import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from helpers.rag import perform_rag, fetch_repos_from_pincone
from helpers.repo import process_repo

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize session states
if "selected_codebase" not in st.session_state:
    st.session_state.selected_codebase = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Modify the dialog to use a more explicit state management approach
@st.dialog("Start a new conversation")
def select_codebase():
    options = fetch_repos_from_pincone()
    
    # Create selectbox with explicit key and callback
    selected = st.selectbox(
        "Select a codebase already indexed to Pinecone:",
        options=options,
        index=options.index(st.session_state.selected_codebase) if st.session_state.selected_codebase in options else 0,
        placeholder="Select codebase",
        key="selected_codebase",
        on_change=main
    )

    github_repo_url = st.text_input("Or paste a public GitHub Link:", placeholder="https://github.com/username/repo")
    process_github = st.button("Index GitHub Repository")
    
    if process_github and github_repo_url:
        with st.spinner('Uploading Repository Data to Pinecone...'):
            process_repo(github_repo_url)
            st.write(f"{github_repo_url} uploaded to Pinecone successfully!")

def main():
    col1, col2 = st.columns([0.70, 0.30], gap="small")

    with col1:
        st.title("💬 Codebase RAG")

    with col2:
        st.write("")
        st.write("")
        open_select_modal = st.button("Select Codebase")

    st.write("This is a chatbot that uses Groq's Llama 3.1 Versatile model to generate responses.")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

    if open_select_modal:
        select_codebase()

    # Always display the currently selected codebase
    if st.session_state.selected_codebase:
        st.write(f"Codebase in use: {st.session_state.selected_codebase}")

    # Chat Section
    if st.session_state.selected_codebase:
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input handling
        if prompt := st.chat_input("Chat..."):
            # Store user message in session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            try:
                # Generate and display assistant response
                llm_response = perform_rag(client, prompt, st.session_state.selected_codebase)
                
                with st.chat_message("assistant"):
                    st.write(llm_response)
                
                # Store assistant message
                st.session_state.messages.append({"role": "assistant", "content": llm_response})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

main()