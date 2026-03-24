from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import redis
import uuid
import os

app = FastAPI()

# Connect to Redis
r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

# Shared directories
UPLOAD_DIR = "/app/shared_volume/uploads"
OUTPUT_DIR = "/app/shared_volume/completed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. API: Receive Image & Queue it
@app.post("/api/enhance/")
async def enhance_image(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{task_id}_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # Put task in queue and set initial status
    r.set(f"status:{task_id}", "processing")
    r.lpush("image_processing_queue", f"{task_id}|{input_path}")
    
    return {"task_id": task_id}

# 2. API: Check Queue Status
@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    status = r.get(f"status:{task_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": status}

# 3. API: Serve the Final Image
@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    status = r.get(f"status:{task_id}")
    if status == "completed":
        # Find the file that starts with enhanced_task_id
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith(f"enhanced_{task_id}"):
                return FileResponse(f"{OUTPUT_DIR}/{filename}")
    raise HTTPException(status_code=404, detail="Image not ready or not found")

# 4. Frontend: Serve the HTML Website
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("/app/frontend/index.html", "r") as f:
        return f.read()