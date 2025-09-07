from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Blob Tracker API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/test-upload/")
async def test_upload(file: UploadFile = File(...)):
    return {"filename": file.filename, "size": len(await file.read())}
