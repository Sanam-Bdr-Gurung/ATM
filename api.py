from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from audio_input import load_audio_bytes
from features import chroma_from_audio
from chord_match import match_chords
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
    raw = await file.read()
    y, sr = load_audio_bytes(raw, sr=22050)
    chroma, times = chroma_from_audio(y, sr)
    labels = match_chords(chroma)
    return {"num_frames": len(labels), "first_labels": labels[:8]}
