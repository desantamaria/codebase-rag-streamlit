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
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

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

LANGUAGE_MAP = {
    '.ts': Language.TS,
    '.tsx': Language.TS,
    '.js': Language.JS,
    '.jsx': Language.JS,
    '.py': Language.PYTHON,
}

def get_language_from_extension(file_name):
    ext = os.path.splitext(file_name)[1]
    return LANGUAGE_MAP.get(ext)

def upload_repo_to_pinecone(file_content, repo_url):
    documents = []

    def process_file(file):
        file_language = get_language_from_extension(file['name'])

        if file_language in LANGUAGE_MAP.values():
            # If the file language is supported, use the splitter
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=file_language, chunk_size=1000, chunk_overlap=100
            )
            chunks = splitter.create_documents([file['content']])
            
            for idx, chunk in enumerate(chunks):
                chunk.metadata = {"source": file['name'], "chunk_index": idx}
                documents.append(chunk)
        else:
            # If the file language is not supported, add the whole file content as a single document
            doc = Document(page_content=file['content'], metadata={"source": file['name'], "chunk_index": 0})
            documents.append(doc)
    
    # Process each file
    for file in file_content:
        process_file(file)

    # Add the documents to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
        index_name="codebase-rag",
        namespace=repo_url
    )
    vectorstore.add_documents(documents)

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
    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace=repo)

    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    file_names = [item['metadata']['source'] for item in top_matches['matches']]
    
    context_summary = "\n\n".join(
        [f"File: {file_name}\nContent:\n{context[:500]}..." for file_name, context in zip(file_names, contexts)]
    )

    augmented_query = f"""
    # Codebase Context:
    {context_summary}

    # Developer Question:
    {query}

    Please provide a response based on the provided context and the specific question.
    """

    system_prompt = f"""
    You are a Senior Software Engineer, specializing in TypeScript, Python, and JavaScript. 
    You are helping a developer understand a codebase. You have access to multiple files and their context.

    Your goal is to:
    1. Provide clear explanations of the code based on the provided context.
    2. Answer both general and specific questions about the repository.
    3. Use all relevant information from the code context and metadata to form your response.
    4. When answering, explain the purpose of the files, functions, or variables if relevant.

    ### How to respond:
    - Always include the file names from the chunk metadata in your response.
    - For general questions, explain the project structure and any high-level files (e.g., README.md).
    - If necessary, summarize or explain large code snippets to fit the response.
    - If necessary, provide code in snippets to improve clarity.
    - Be concise but thorough when explaining technical details.

    # Provided context:
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