import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

app = FastAPI(title="VideoAgent API")

# Allow the React dev frontend (served from a local file or localhost) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store: session_id -> {"rag_chain": ..., "transcript": ...}
# NOTE: this resets whenever the server restarts, and is not safe for multiple
# concurrent users in production -- it's intentionally simple for a demo/showcase.
SESSIONS: dict = {}


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/process")
async def process_video(
    language: str = Form("english"),
    youtube_url: str = Form(None),
    file: UploadFile = File(None),
):
    if not youtube_url and not file:
        raise HTTPException(status_code=400, detail="Provide either a YouTube URL or a file upload.")

    if youtube_url:
        source = youtube_url
    else:
        # Save the uploaded file to disk so process_input can treat it like any local file.
        dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source = dest_path

    try:
        chunks = process_input(source)
        transcript = transcribe_all(chunks, language=language)

        # These five calls are all independent (each just takes the transcript),
        # so run them concurrently instead of one after another. Since these are
        # network-bound API calls to Mistral, threads work well here even though
        # Python has a GIL -- the GIL is released while waiting on network I/O.
        with ThreadPoolExecutor(max_workers=5) as executor:
            title_future = executor.submit(generate_title, transcript)
            summary_future = executor.submit(summarize, transcript)
            action_items_future = executor.submit(extract_action_items, transcript)
            decisions_future = executor.submit(extract_key_decisions, transcript)
            questions_future = executor.submit(extract_questions, transcript)

            title = title_future.result()
            summary = summary_future.result()
            action_items = action_items_future.result()
            decisions = decisions_future.result()
            questions = questions_future.result()

        session_id = str(uuid.uuid4())
        rag_chain = build_rag_chain(transcript, collection_name=f"meeting_{session_id}")

        SESSIONS[session_id] = {"rag_chain": rag_chain, "transcript": transcript}

        return {
            "session_id": session_id,
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please process a video first.")

    try:
        answer = ask_question(session["rag_chain"], req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))