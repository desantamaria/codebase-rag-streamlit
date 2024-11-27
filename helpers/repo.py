from git import Repo
import os
import tempfile
from helpers.rag import upload_repo_to_pinecone

SUPPORTED_EXTENSIONS= [".py", ".js", ".tsx", ".jsx", ".ts", ".java, .cpp"]

IGNORED_DIRS = [".git", ".github", "dist", "__pycache__", ".next", ".env", ".vscode", "node_modules", ".venv", "h5"]

def clone_repo(repo_url):
    repo_name = repo_url.split("/")[-1]
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, repo_name)
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

def process_repo(repo_url):
    path = clone_repo(repo_url)
    file_content = get_main_files_content(path)
    upload_repo_to_pinecone(file_content, repo_url)

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        rel_path = os.path.relpath(file_path, repo_path)
        return {
            "name": rel_path,
            "content": content,
        }
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def get_main_files_content(repo_path: str):
    """
    Get content of supported code files from the local repository.


    Args:
        repo_path: Path to the local repository


    Returns:
        List of dictionaries containing file names and contents
    """
    files_content = []


    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue


            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)


    except Exception as e:
        print(f"Error reading repository: {str(e)}")


    return files_content