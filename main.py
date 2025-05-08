from fastapi import FastAPI, HTTPException
from evaluator import process_submission  # Assuming your method is process_submission

app = FastAPI(title="Assignment Auto-Evaluator", version="1.0")

# Static values for sheet_name and credentials_json as per your original setup
sheet_name = 'Assignment Submission Form'  
credentials_json = 'credentials.json'

@app.get("/evaluate")
def evaluate_submissions():
    try:
        # Only call the process_submission function here when the endpoint is hit
        result = process_submission(sheet_name, credentials_json)
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Assignment Auto-Evaluator is running. Use GET /evaluate to start."}
