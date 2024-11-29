from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain.schema import Document
from pinecone import Pinecone
import google.generativeai as genai
from PIL import Image


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your Pinecone index
pinecone_index = pc.Index("codebase-rag")
vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

def upload_repo_to_pinecone(file_content, repo_url):
    # Insert the codebase embeddings into Pinecone
    documents = []

    for file in file_content:
        doc = Document (
            page_content=f"{file['name']}\n{file['content']}",
            metadata={"source": file[ 'name']}
        )
        documents. append (doc)
        
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        index_name="codebase-rag",
        namespace=repo_url
    )

def fetch_repos_from_pincone():
    try:
        index_stats = pinecone_index.describe_index_stats()        
        all_namespaces = list(index_stats.get('namespaces', {}).keys())
        return all_namespaces
    except Exception as e:
        print(f"An error occurred: {e}")

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

def perform_rag(model, query, repo, selected_model, image=None):
    raw_query_embedding = get_huggingface_embeddings(query)
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=repo)

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_query = "\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n\n\n\n\nMY QUESTION:\n" + query

    system_prompt = f"""
    You are a Senior Software Engineer, specializing in TypeScript.
    Answer questions based on the provided code context. Always use all available information to form your response.
    """

    if selected_model == "Groq's Llama 3.1":
        # Groq's Llama 3.1 API call
        llm_response = model.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )
        return llm_response.choices[0].message.content

    elif selected_model == "Google Gemini":
        # Google Gemini API call with optional image input
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        if image:
            img = Image.open(image)
            llm_response = gemini_model.generate_content([augmented_query, img])
        else:
            llm_response = gemini_model.generate_content([augmented_query])
        
        return llm_response.text

    else:
        raise ValueError(f"Unknown model selected: {selected_model}")