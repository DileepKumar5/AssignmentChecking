import os
import nbformat
import json
import shutil  # For file moving operations
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Extract code from notebook
def extract_code_from_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    return "\n\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'code')

# Extract rubric from notebook (markdown/text cells)
def extract_rubric_from_notebook(rubric_notebook_path):
    nb = nbformat.read(rubric_notebook_path, as_version=4)
    rubric_text = "\n\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'markdown')
    return [Document(page_content=rubric_text)]

# Set up RAG using rubric
def setup_rag_from_notebook(rubric_notebook_path):
    documents = extract_rubric_from_notebook(rubric_notebook_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    return db.as_retriever(search_kwargs={"k": 3})

# Evaluate code
def evaluate_code(code_str, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    prompt = f"""
    You are an expert Python teacher. Evaluate this student's code based on the rubric.
    Provide a score (0–100) and feedback. Feedback should not be long, just 3 or 4 sentences.

    ```python
    {code_str}
    ```

    Return as valid JSON:
    {{
      "score": <integer>,
      "feedback": "<feedback>"
    }}
    """
    return qa_chain.run({"query": prompt})

# Save result to file
def save_result(result_str, submission_path):
    base_name = os.path.splitext(os.path.basename(submission_path))[0]
    output_path = f"marked/{base_name}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result_str)
    print(f"[💾] Result saved to {output_path}")

# Move the submitted notebook to the destination folder
def move_submission_to_destination(submission_path, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # Create the folder if it doesn't exist
    file_name = os.path.basename(submission_path)
    destination_path = os.path.join(destination_folder, file_name)
    
    # Move the file
    shutil.move(submission_path, destination_path)
    print(f"[📦] Moved {file_name} to {destination_folder}")

# Main
def main():
    source_path = "Submissions/Assignment_7(Hamza khan).ipynb"
    question_path = "Questions/rubric.ipynb"
    destination_folder = "Destination"

    print("[🔍] Extracting code...")
    code = extract_code_from_notebook(notebook_path)

    print("[📚] Setting up rubric context...")
    retriever = setup_rag_from_notebook(rubric_notebook_path)

    print("[🤖] Evaluating code...")
    result = evaluate_code(code, retriever)

    print("\n✅ Evaluation Result:")
    print(result)

    # Save the result to the marked folder
    save_result(result, notebook_path)

    # Move the notebook to the destination folder and delete from the source
    move_submission_to_destination(notebook_path, destination_folder)

if __name__ == "__main__":
    main()
