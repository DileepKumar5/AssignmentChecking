import os
import nbformat
import shutil
import io
import re
import json
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
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

# Authenticate with Google Drive and Sheets
def authenticate_google_services(credentials_json):
    credentials = Credentials.from_service_account_file(
        credentials_json,
        scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    )
    
    # Google Drive service
    drive_service = build('drive', 'v3', credentials=credentials)
    
    # Google Sheets service
    gc = gspread.authorize(credentials)
    
    return drive_service, gc

# Extract file ID from Google Form URL
def extract_file_id(file_url):
    try:
        if "/d/" in file_url:
            file_id = file_url.split("/d/")[1].split("/")[0]
        elif "open?id=" in file_url:
            file_id = file_url.split("open?id=")[1].split("&")[0]
        else:
            raise ValueError("File URL is not in the expected format")
        return file_id
    except IndexError:
        raise ValueError(f"Invalid URL format: {file_url}")

# Download file from Google Drive
def download_file_from_drive(drive_service, file_id, destination_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    print(f"Downloaded file to {destination_path}")

# Extract code from notebook
def extract_code_from_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    return "\n\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'code')

# Extract rubric from notebook
def extract_rubric_from_notebook(rubric_notebook_path):
    # Check if the file exists
    if not os.path.exists(rubric_notebook_path):
        raise FileNotFoundError(f"Rubric notebook not found: {rubric_notebook_path}")
    
    nb = nbformat.read(rubric_notebook_path, as_version=4)
    rubric_text = "\n\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'markdown')
    return [Document(page_content=rubric_text)]

# Set up RAG from rubric notebook
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
# Save result to file
def save_result(result_str, submission_path):
    base_name = os.path.splitext(os.path.basename(submission_path))[0]
    output_path = f"marked/{base_name}.txt"
    
    # Ensure 'marked/' directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
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

# Sanitize the timestamp
def sanitize_timestamp(timestamp):
    return re.sub(r'[\\/:*?"<>|]', '_', timestamp)

# Check if the URL is valid
def is_valid_url(url):
    return "drive.google.com" in url and ("open?id=" in url or "/d/" in url)

# Fetch submission URLs from the Sheet
def fetch_submission_urls(sheet_name, credentials_json):
    _, gc = authenticate_google_services(credentials_json)
    sheet = gc.open(sheet_name).sheet1
    
    headers = sheet.row_values(1)
    print("Headers found in sheet:", headers)

    rows = sheet.get_all_records()

    submission_data = []
    for row in rows:
        if row.get("Status", "").strip().lower() == "checked":
            continue  # Skip already checked submissions

        name = row.get("Name")
        timestamp = row.get("Timestamp")
        file_url = row.get("File Upload")
        assignment = row.get("Assignment")

        if file_url and is_valid_url(file_url):
            submission_data.append((file_url, name, timestamp, assignment))

    print(f"✅ {len(submission_data)} submissions to process.")
    return submission_data


def update_sheet_with_results(sheet_name, credentials_json, timestamp, name, marks, feedback):
    _, gc = authenticate_google_services(credentials_json)
    sheet = gc.open(sheet_name).sheet1

    all_records = sheet.get_all_records()
    
    for idx, row in enumerate(all_records, start=2):  # start=2 because header is row 1
        if row["Name"].strip() == name.strip() and row["Timestamp"].strip() == timestamp.strip():
            sheet.update_cell(idx, 6, marks)     # Column F: Marks
            sheet.update_cell(idx, 7, feedback)  # Column G: Feedback
            sheet.update_cell(idx, 8, "Checked") # Column H: Status
            print(f"[✅] Sheet updated for {name} at row {idx}")
            break
    else:
        print(f"[⚠️] Could not find a matching row for {name} at {timestamp}")


# Main function to process the submission
def process_submission(sheet_name, credentials_json):
    folder_path = "studentsubmission"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    submissions = fetch_submission_urls(sheet_name, credentials_json)
    drive_service, gc = authenticate_google_services(credentials_json)
    sheet = gc.open(sheet_name).sheet1  # We’ll need it later for updates
    
    # Set up RAG for each assignment dynamically based on assignment names
    for i, (file_url, student_name, timestamp, assignment_name) in enumerate(submissions):
        assignment_name = assignment_name.strip()

        sanitized_timestamp = sanitize_timestamp(timestamp)
        filename = f"{student_name}_{sanitized_timestamp}.ipynb"
        destination_path = os.path.join(folder_path, filename)
        
        file_id = extract_file_id(file_url)
        download_file_from_drive(drive_service, file_id, destination_path)
        
        # Dynamically construct the rubric notebook path based on assignment name
        rubric_notebook_path = f"Questions/{assignment_name}.ipynb"
        print(f"Using rubric notebook: {rubric_notebook_path}")
        
        if not os.path.exists(rubric_notebook_path):
            print(f"❌ Error: The rubric notebook '{rubric_notebook_path}' does not exist. Skipping this assignment.")
            continue
        
        # Set up RAG retriever from the rubric notebook
        retriever = setup_rag_from_notebook(rubric_notebook_path)
        
        # Extract code from the notebook and evaluate
        code = extract_code_from_notebook(destination_path)
        result = evaluate_code(code, retriever)

        # ✅ Parse JSON result
        try:
            result_json = json.loads(result)
            marks = result_json.get("score", 0)
            feedback = result_json.get("feedback", "No feedback provided.")
        except json.JSONDecodeError:
            marks = 0
            feedback = "❌ Failed to parse evaluation result."
            print("[⚠️] JSON parsing error in evaluation result")
        
        # ✅ Update the Google Sheet
        try:
            cell = sheet.find(timestamp)
            row_number = cell.row
            sheet.update(f"F{row_number}", marks)       # Marks
            sheet.update(f"G{row_number}", feedback)    # Feedback
            sheet.update(f"H{row_number}", "Checked")   # Status
            print(f"[✅] Sheet updated for {student_name} at row {row_number}")
        except Exception as e:
            print(f"[⚠️] Failed to update sheet for {student_name}: {e}")

        # Save the result and move the file
        save_result(result, destination_path)
        move_submission_to_destination(destination_path, "checked")
    
    return "All submissions processed."


# Example usage
sheet_name = 'Assignment Submission Form'  
credentials_json = 'credentials.json'

# Call the main function
result = process_submission(sheet_name, credentials_json)
print(result)
