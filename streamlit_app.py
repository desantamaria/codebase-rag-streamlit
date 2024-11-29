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
    st.session_state.selected_model = "Groq's Llama 3.1"


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

def main():
    st.title("💬 Codebase RAG Chat")

    if st.session_state.selected_codebase:
        st.write(f"Codebase in use: {st.session_state.selected_codebase}")
    else:
        st.warning("No codebase selected. Please select or index a new one.")

    # Chat section
    if st.session_state.selected_codebase:
        # Display previous messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        image = None
        if not st.session_state.selected_model == "Groq's Llama 3.1":
            # Optional image input
            image = st.file_uploader("Upload an image for multimodal input: (Optional)", type=["png", "jpg", "jpeg"])

        # Display preview of image uploaded
        if image:
            st.image(image, caption="Uploaded Image")
            st.write(f"File type: {image.type}, Name: {image.name}, Size: {image.size}")
            st.success("Image uploaded successfuly!")
            
        # Chat input
        prompt = st.chat_input("Chat...")
            
        # Only run if prompt is provided
        if prompt:
            # Switch between models based on user selection
            if st.session_state.selected_model == "Groq's Llama 3.1":
                llm_response = perform_rag(groq_client, prompt, st.session_state.selected_codebase, "Groq's Llama 3.1")
            else:
                llm_response = perform_rag(genai, prompt, st.session_state.selected_codebase, "Google Gemini", image=image)

            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            with st.chat_message("assistant"):
                st.markdown(llm_response)
        elif image and not prompt:
            st.warning("Please provide a text prompt along with the image.")

# Sidebar for codebase and model selection
with st.sidebar:
    st.header("Codebase Selection")
    select_codebase()
    st.header("AI Model Selection")
    select_model()

# Run the main function
if __name__ == "__main__":
    main()

