import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import tempfile
from github import Github, Repository
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone
from git import Repo

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(GROQ_API_KEY)

# Show title and description.
st.title("💬 Codebase RAG")
st.write(
    "This is a chatbot that uses Groq's Llama 3.1 Versatile model to generate responses."
)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
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
    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
    )

    # Write the response to the chat using `st.write`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write(llm_response.choices[0].message.content)
    st.session_state.messages.append({"role": "assistant", "content": response})
