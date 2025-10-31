from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ChordAssist API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}



@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}