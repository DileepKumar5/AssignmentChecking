import os
import nbformat
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access the variables
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def send_email(to_email, subject, body):
    try:
        # Create the MIME object with HTML content type
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = to_email

        # Create the plain-text and HTML versions of the message
        part1 = MIMEText(body, "html")  # Specify 'html' to ensure HTML rendering
        msg.attach(part1)

        # Send the email
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"[📧] Feedback sent to {to_email}")
    except Exception as e:
        print(f"[⚠️] Failed to send email to {to_email}: {e}")


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


# Download notebook content directly to memory (no file saving)
def download_notebook_content(drive_service, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return nbformat.read(fh, as_version=4)  # return notebook object


# Extract rubric from notebook
def extract_rubric_from_notebook(rubric_notebook_path):
    if not os.path.exists(rubric_notebook_path):
        raise FileNotFoundError(f"Rubric notebook not found: {rubric_notebook_path}")
    
    nb = nbformat.read(rubric_notebook_path, as_version=4)
    rubric_text = "\n\n".join(cell['source'] for cell in nb.cells if cell.cell_type == 'markdown')
    return [Document(page_content=rubric_text)]


# Set up RAG from rubric notebook
def setup_rag_from_notebook(rubric_notebook_path):
    documents = extract_rubric_from_notebook(rubric_notebook_path)
    full_rubric_text = documents[0].page_content  # Extract the full text for question

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    db = Chroma.from_documents(docs, OpenAIEmbeddings())
    return db.as_retriever(search_kwargs={"k": 3}), full_rubric_text


# Evaluate code
def evaluate_code(code_str, retriever, assignment_question):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    prompt = f"""
    You are a senior Data Science instructor reviewing a student's assignment submission.

    You have both the original assignment question and the student's submission.

    1. If the student has submitted **only Markdown** (i.e., no Python code at all), or if they have **copied the assignment question without answering**, then:
        - Assign a score of 0.
        - Write clear feedback using a friendly but firm tone like a teacher, saying things like:
            - "You did not implement any code."
            - "Your submission is just a copy of the question or explanation."
            - "To receive credit, you must submit working Python code, including model training, evaluation, and comparison."
        - End the feedback by encouraging the student to try again.

    2. If the student has written **actual Python code**, assess their work in terms of:
        - Correct use of data science and machine learning techniques
        - Code quality, logic, structure
        - Use of proper libraries (e.g., pandas, sklearn, tensorflow, matplotlib, etc.)
        - Whether the steps align with the assignment’s goals (e.g., training ANN vs CNN on MNIST)

    3. Provide helpful feedback using a teaching tone:
        - Point out what **you did well**
        - Point out **what’s missing or wrong**
        - Suggest how **you can improve**
        - Provide any best practices or tips

    4. Give a score based on:
        - 100: Excellent work with no issues
        - 90–99: Great work with minor issues
        - 80–89: Good work with some improvements needed
        - 0: If there is no code, or it’s just Markdown/copy-paste of the question

    Return your result strictly as a **valid JSON**:
    {{
    "score": <integer between 0 and 100>,
    "feedback": "<helpful, direct feedback using 'you'>"
    }}

    Here is the assignment question:

    {assignment_question}

    Here is the student's submission:

    ```python
    {code_str}
    ```
    """
    return qa_chain.run({"query": prompt})


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
            email = row.get("Email Address")
            submission_data.append((file_url, name, timestamp, assignment, email))

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
    submissions = fetch_submission_urls(sheet_name, credentials_json)
    drive_service, gc = authenticate_google_services(credentials_json)
    sheet = gc.open(sheet_name).sheet1  # We’ll need it later for updates
    
    # Set up RAG for each assignment dynamically based on assignment names
    for i, (file_url, student_name, timestamp, assignment_name, student_email) in enumerate(submissions):
        sanitized_timestamp = sanitize_timestamp(timestamp)
        
        file_id = extract_file_id(file_url)
        notebook = download_notebook_content(drive_service, file_id)

        # Dynamically construct the rubric notebook path based on assignment name
        rubric_notebook_path = f"Questions/{assignment_name}.ipynb"
        print(f"Using rubric notebook: {rubric_notebook_path}")
        
        if not os.path.exists(rubric_notebook_path):
            print(f"❌ Error: The rubric notebook '{rubric_notebook_path}' does not exist. Skipping this assignment.")
            continue
        
        # Set up RAG retriever from the rubric notebook
        retriever, assignment_question = setup_rag_from_notebook(rubric_notebook_path)

        # Extract code from the notebook and evaluate
        code = "\n\n".join(cell['source'] for cell in notebook.cells if cell.cell_type == 'code')
        result = evaluate_code(code, retriever, assignment_question)

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

        # Send the email with feedback and marks
        email_subject = f"Feedback for {assignment_name} - {marks}/100"
        email_body = f"""<html>
            <body>
                <p>Dear <strong>{student_name}</strong>,</p>

                <p>I hope this message finds you well.</p>

                <p>I have reviewed your submission for the assignment '<strong>{assignment_name}</strong>' and would like to provide the following evaluation:</p>

                <table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
                    <tr>
                        <th style="text-align: left;">Score</th>
                        <td>{marks}/100</td>
                    </tr>
                    <tr>
                        <th style="text-align: left;">Feedback</th>
                        <td>{feedback}</td>
                    </tr>
                </table>

                <p>If you have any questions or would like further clarification, please feel free to reach out to me.</p>

                <p><strong>Best regards,</strong><br>
                Dileep Kumar<br>
                Instructor | DataCrumbs Team</p>
            </body>
        </html>"""

        send_email(student_email, email_subject, email_body)
        print(f"[📧] Feedback sent to {student_email}")

    return "All submissions processed."


# Example usage
sheet_name = 'Assignment Submission Form'  
credentials_json = 'credentials.json'

# Call the main function
result = process_submission(sheet_name, credentials_json)
print(result)
