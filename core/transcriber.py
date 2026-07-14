import os
import requests
import whisper
from pydub import AudioSegment
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_PIECE_SECONDS = 25  # stay safely under Sarvam's 30-second REST limit
SARVAM_MAX_WORKERS = 5  # concurrent uploads; keep modest to avoid rate limits

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe", fp16=False)
    return result["text"].strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send a single short (<=30s) audio file to Sarvam's speech-to-text-translate endpoint."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=300,
        )
        response.raise_for_status()
        return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam's REST endpoint only accepts audio up to 30 seconds, so this splits
    the (potentially several-minute-long) chunk into small pieces and sends
    several pieces to Sarvam concurrently (instead of one at a time), then
    stitches the results back together in the original order.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment/.env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    # Export all pieces to disk first.
    piece_paths = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start:start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        piece_paths.append(piece_path)

    results = [None] * len(piece_paths)

    try:
        with ThreadPoolExecutor(max_workers=SARVAM_MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(_send_to_sarvam, path): i
                for i, path in enumerate(piece_paths)
            }
            completed = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                results[i] = future.result()
                completed += 1
                print(f"  -> Sarvam piece {completed}/{total_pieces} done")
    finally:
        for path in piece_paths:
            if os.path.exists(path):
                os.remove(path)

    return " ".join(results).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
      - english  -> Whisper (local model)
      - hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription completed")
    return full_transcript.strip()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.audio_processor import process_input

    test_url = "https://www.youtube.com/watch?v=d-CuF6dlqLg"
    chunks = process_input(test_url)

    transcript = transcribe_all(chunks, language="english")
    print("\n--- Full Transcript ---\n")
    print(transcript)