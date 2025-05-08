import io
import nbformat
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import re
import os


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
        # Handle both types of URLs: open?id= and file/d/
        if "/d/" in file_url:
            file_id = file_url.split("/d/")[1].split("/")[0]
        elif "open?id=" in file_url:
            file_id = file_url.split("open?id=")[1].split("&")[0]
        else:
            raise ValueError("File URL is not in the expected format")
        return file_id
    except IndexError:
        raise ValueError(f"Invalid URL format: {file_url}")

# Download the file from Google Drive
def download_file_from_drive(drive_service, file_id, destination_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    print(f"Downloaded file to {destination_path}")

# Read the .ipynb notebook content
def read_ipynb(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Print out the content of the notebook
    cells = notebook.cells
    print("\nNotebook Content:\n")
    for cell in cells:
        if cell.cell_type == 'code':
            print("Code Cell:")
            print(cell.source)
        elif cell.cell_type == 'markdown':
            print("Markdown Cell:")
            print(cell.source)
    
    return notebook

# Check if the URL is valid (new method to handle the 'open?id=' format)
def is_valid_url(url):
    return "drive.google.com" in url and ("open?id=" in url or "/d/" in url)

# Fetch Submission URLs from the Sheet
def fetch_submission_urls(sheet_name, credentials_json):
    # Authenticate and access Google Sheets
    _, gc = authenticate_google_services(credentials_json)
    
    # Open the Google Sheet by name
    sheet = gc.open(sheet_name).sheet1
    
    # Fetch headers to get column indices dynamically
    headers = sheet.row_values(1)
    name_col_index = headers.index("Name") + 1  # Column index for "Name"
    timestamp_col_index = headers.index("Timestamp") + 1  # Column index for "Timestamp"
    file_url_col_index = headers.index("File Upload") + 1  # Column index for "File Upload"

    # Fetch all rows of data
    rows = sheet.get_all_records()

    # Extract the relevant data: Name, Timestamp, and File URL
    names = [row["Name"] for row in rows]
    timestamps = [row["Timestamp"] for row in rows]
    file_urls = [row["File Upload"] for row in rows]
    
    # Filter out invalid or empty URLs
    valid_urls = [url for url in file_urls if url and is_valid_url(url)]
    
    # Debugging: print valid URLs, names, and timestamps
    print("Valid Submission URLs:", valid_urls)
    print("Names:", names)
    print("Timestamps:", timestamps)
    
    return valid_urls, names, timestamps

# Function to sanitize the timestamp
def sanitize_timestamp(timestamp):
    # Replace invalid characters with underscores (_)
    return re.sub(r'[\\/:*?"<>|]', '_', timestamp)

# Main function to process the submission
# Main function to process the submission
def process_submission(sheet_name, credentials_json):
    # Create a folder to save the files if it doesn't exist
    folder_path = "studentsubmission"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Fetch file URLs from Google Sheet
    submission_urls, names, timestamps = fetch_submission_urls(sheet_name, credentials_json)
    
    drive_service, _ = authenticate_google_services(credentials_json)
    
    for i, file_url in enumerate(submission_urls):
        student_name = names[i]  # Get student's name from the sheet
        timestamp = timestamps[i]  # Get timestamp from the sheet
        
        # Sanitize timestamp to remove invalid characters for filenames
        sanitized_timestamp = sanitize_timestamp(timestamp)
        
        # Generate the filename as Name_Timestamp.ipynb
        filename = f"{student_name}_{sanitized_timestamp}.ipynb"
        
        # Define the full path to save the file in the 'studentsubmission' folder
        destination_path = os.path.join(folder_path, filename)
        
        print(f"Processing {file_url} for {student_name} at {timestamp}...")
        
        file_id = extract_file_id(file_url)
        download_file_from_drive(drive_service, file_id, destination_path)
        read_ipynb(destination_path)
    
    return "All submissions processed."

# Example usage
sheet_name = 'Assignment Submission Form'  # Name of your Google Sheet with submission URLs
credentials_json = 'credentials.json'  # Path to your credentials file

# Call the main function to process all submissions
result = process_submission(sheet_name, credentials_json)
print(result)
