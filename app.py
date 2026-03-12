"""
PDF-to-Audiobook Flask App
==========================
Upload a PDF, pick pages and a TTS backend, download an MP3.

Backends:
  - polly   : Amazon Polly (requires AWS credentials)
  - piper   : Local Piper TTS (piper-tts Python lib → CLI fallback)
  - huggingface : HF Transformers TTS pipeline

All configuration lives in .env — see .env for docs.
"""

import io
import os
import re
import time
import wave
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env before anything reads os.getenv

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    flash,
    redirect,
    url_for,
)
from pypdf import PdfReader
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Configuration — all values come from .env (see .env for docs)
# ---------------------------------------------------------------------------

CONFIG = {
    # General
    "DEFAULT_BACKEND": os.getenv("TTS_BACKEND", "piper"),
    # Amazon Polly
    "POLLY_VOICE_ID": os.getenv("POLLY_VOICE_ID", "Joanna"),
    "POLLY_ENGINE": os.getenv("POLLY_ENGINE", "neural"),
    "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
    # Piper
    "PIPER_MODEL": str(
        (Path(__file__).parent / os.getenv("PIPER_MODEL", "models/en_US-amy-medium.onnx")).resolve()
    ),
    "PIPER_BINARY": os.getenv("PIPER_BINARY", "piper"),
    "PIPER_LENGTH_SCALE": float(os.getenv("PIPER_LENGTH_SCALE", "1.0")),
    "PIPER_NOISE_SCALE": float(os.getenv("PIPER_NOISE_SCALE", "0.667")),
    "PIPER_NOISE_W_SCALE": float(os.getenv("PIPER_NOISE_W_SCALE", "0.8")),
    "PIPER_SENTENCE_SILENCE": float(os.getenv("PIPER_SENTENCE_SILENCE", "0.4")),
    # Hugging Face
    "HF_MODEL": os.getenv("HF_MODEL", "facebook/mms-tts-eng"),
}

MAX_UPLOAD_MB = 50
MAX_TEXT_CHARS = 500_000  # safety cap — ~200 pages of text
POLLY_CHUNK_CHARS = 2800  # safe limit per Polly SynthesizeSpeech call
ALLOWED_EXTENSIONS = {".pdf"}
PDF_MAGIC = b"%PDF"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (heavy models loaded once)
# ---------------------------------------------------------------------------

_piper_voice = None
_hf_pipeline = None


def _get_piper_voice(model_path: str):
    """Load PiperVoice once and cache it."""
    global _piper_voice
    if _piper_voice is None or _piper_voice._model_path != model_path:
        from piper import PiperVoice

        log.info("Loading Piper model: %s", model_path)
        _piper_voice = PiperVoice.load(model_path)
        _piper_voice._model_path = model_path  # tag for cache check
    return _piper_voice


def _get_hf_pipeline(model_name: str):
    """Load HF TTS pipeline once and cache it."""
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline as hf_pipeline

        log.info("Loading HF TTS model: %s", model_name)
        _hf_pipeline = hf_pipeline("text-to-speech", model=model_name)
    return _hf_pipeline


# ---------------------------------------------------------------------------
# Text preprocessing — makes TTS output sound like an audiobook
# ---------------------------------------------------------------------------


def preprocess_text_for_speech(text: str) -> str:
    """Clean up PDF-extracted text for more natural TTS output.

    - Collapses PDF junk whitespace
    - Normalizes punctuation for better pauses
    - Adds sentence-ending punctuation where missing (so TTS engines pause)
    - Converts paragraph breaks into explicit pause markers
    """
    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize common PDF extraction artifacts (ligatures)
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = text.replace("\ufb00", "ff")
    text = text.replace("\ufb03", "ffi")
    text = text.replace("\ufb04", "ffl")
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u2013\u2014]", " -- ", text)
    text = text.replace("\u2026", "...")

    # Remove stray hyphens from line-break word splits: "impor-\ntant" -> "important"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Collapse 3+ newlines into double (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Lines that look like headings (ALL CAPS, 5+ chars) — add pause after
    text = re.sub(
        r"^([A-Z][A-Z\s\d]{4,})$",
        r"\1.\n",
        text,
        flags=re.MULTILINE,
    )

    # Paragraph breaks -> period + newline (ensures TTS sees sentence boundary)
    text = re.sub(r"\n\n+", ".\n\n", text)

    # Single newlines mid-paragraph -> space (PDF line wrapping)
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)

    # Clean up doubled periods from above transforms
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\.\s*\.", ".", text)

    # Ensure text ends with punctuation
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    return text


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def parse_page_range(range_str: str, total_pages: int) -> list[int]:
    """Parse a page-range string into 0-based page indices.

    Accepts: "all", "5", "1-10", "3-3".
    Returns sorted list of valid 0-based indices.
    Raises ValueError on bad input.
    """
    s = range_str.strip().lower()
    if s == "all":
        return list(range(total_pages))

    m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", s)
    if not m:
        raise ValueError(f"Invalid page range: '{range_str}'")

    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start

    if start < 1 or end < start:
        raise ValueError(f"Invalid page range: '{range_str}'")
    if end > total_pages:
        raise ValueError(
            f"Page {end} exceeds document length ({total_pages} pages)"
        )

    return list(range(start - 1, end))  # convert to 0-based


def extract_text_from_pdf(file_stream, page_range_str: str) -> tuple[str, str]:
    """Extract and preprocess text from selected PDF pages."""
    reader = PdfReader(file_stream)
    total = len(reader.pages)
    indices = parse_page_range(page_range_str, total)

    parts: list[str] = []
    for i in indices:
        text = reader.pages[i].extract_text() or ""
        parts.append(text)

    raw = "\n".join(parts)
    cleaned = preprocess_text_for_speech(raw)

    if page_range_str.strip().lower() == "all":
        label = "all"
    elif len(indices) == 1:
        label = f"p{indices[0]+1}"
    else:
        label = f"p{indices[0]+1}-{indices[-1]+1}"

    return cleaned, label


def _slug_token(value: str, max_len: int = 50) -> str:
    """Convert text to a compact filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    token = re.sub(r"-+", "-", token).strip("-._")
    return (token[:max_len] or "na")


def build_output_filename(
    pdf_filename: str,
    page_label: str,
    backend: str,
    provider_detail: str,
) -> str:
    """Create stable output names:

    <ISO-UTC>_<book>_<pages>_<backend>_<voice-or-model>.mp3
    """
    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    book = _slug_token(Path(pdf_filename).stem, max_len=60)
    pages = _slug_token(page_label, max_len=24)
    backend_token = _slug_token(backend, max_len=16)
    detail = _slug_token(provider_detail, max_len=80)
    return f"{iso_ts}_{book}_{pages}_{backend_token}_{detail}.mp3"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chars: int = POLLY_CHUNK_CHARS) -> list[str]:
    """Split text into chunks on whitespace boundaries, each ≤ max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        # find last whitespace before the limit
        cut = text.rfind(" ", 0, max_chars)
        if cut == -1:
            cut = max_chars  # no whitespace found — hard cut
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


# ---------------------------------------------------------------------------
# TTS backends
# ---------------------------------------------------------------------------


# Polly per-million-character rates (USD) — update if AWS pricing changes
_POLLY_RATES = {
    "standard":   4.00,
    "neural":     16.00,
    "long-form":  100.00,
    "generative": 30.00,
}


def synthesize_polly(text: str, voice_id: str, engine: str) -> tuple[io.BytesIO, int]:
    """Synthesize text → MP3 via Amazon Polly with SSML pauses.

    Returns (mp3_buffer, total_chars_billed) where total_chars_billed is the
    sum of RequestCharacters reported by Polly across all chunks.
    """
    import boto3

    polly = boto3.client("polly", region_name=CONFIG["AWS_REGION"])

    def _to_ssml(chunk: str) -> str:
        """Wrap a text chunk in SSML with pauses at paragraph boundaries."""
        chunk = chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Paragraph breaks → 600ms pause
        chunk = chunk.replace("\n\n", '<break time="600ms"/>')
        # Single newlines → small breath pause
        chunk = chunk.replace("\n", '<break time="250ms"/>')
        return f"<speak>{chunk}</speak>"

    chunks = chunk_text(text, POLLY_CHUNK_CHARS - 200)  # headroom for SSML tags
    log.info("Polly: %d chunks, voice=%s, engine=%s", len(chunks), voice_id, engine)

    combined = AudioSegment.empty()
    total_chars_billed = 0
    for i, chunk in enumerate(chunks):
        ssml = _to_ssml(chunk)
        resp = polly.synthesize_speech(
            Text=ssml,
            TextType="ssml",
            OutputFormat="mp3",
            VoiceId=voice_id,
            Engine=engine,
        )
        chars_this_chunk = resp.get("RequestCharacters", len(chunk))
        total_chars_billed += chars_this_chunk
        audio_bytes = resp["AudioStream"].read()
        segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        combined += segment
        log.info("  chunk %d/%d done (%d chars billed)", i + 1, len(chunks), chars_this_chunk)

    buf = io.BytesIO()
    combined.export(buf, format="mp3", bitrate="128k")
    buf.seek(0)
    rate = _POLLY_RATES.get(engine.lower(), _POLLY_RATES["neural"])
    cost_usd = total_chars_billed * rate / 1_000_000
    log.info("Polly billed: %d chars, engine=%s, estimated cost $%.6f", total_chars_billed, engine, cost_usd)
    return buf, total_chars_billed


def synthesize_piper(
    text: str,
    model_path: str,
    length_scale: float | None = None,
    noise_scale: float | None = None,
    noise_w_scale: float | None = None,
    sentence_silence: float | None = None,
) -> io.BytesIO:
    """Synthesize text → MP3 via Piper TTS with prosody controls.

    Uses the Python API's synthesize() iterator so we can inject silence
    between sentences for natural audiobook pacing.
    """
    ls = length_scale if length_scale is not None else CONFIG["PIPER_LENGTH_SCALE"]
    ns = noise_scale if noise_scale is not None else CONFIG["PIPER_NOISE_SCALE"]
    nws = noise_w_scale if noise_w_scale is not None else CONFIG["PIPER_NOISE_W_SCALE"]
    ss = sentence_silence if sentence_silence is not None else CONFIG["PIPER_SENTENCE_SILENCE"]

    wav_buf = io.BytesIO()
    try:
        from piper.config import SynthesisConfig

        voice = _get_piper_voice(model_path)
        log.info(
            "Piper (Python API): model=%s, text_len=%d, "
            "length_scale=%.2f, noise_scale=%.3f, noise_w=%.2f, sentence_silence=%.2fs",
            model_path, len(text), ls, ns, nws, ss,
        )

        syn_config = SynthesisConfig(
            length_scale=ls,
            noise_scale=ns,
            noise_w_scale=nws,
        )

        sample_rate = voice.config.sample_rate
        silence_samples = int(sample_rate * ss)
        silence_bytes = b"\x00" * (silence_samples * 2)  # 16-bit = 2 bytes/sample

        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            for i, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
                if i > 0 and ss > 0:
                    wf.writeframes(silence_bytes)
                wf.writeframes(audio_chunk.audio_int16_bytes)

    except Exception as e:
        log.warning("Piper Python API failed (%s), falling back to CLI", e)
        wav_buf = _synthesize_piper_cli(text, model_path, ls, ss)

    wav_buf.seek(0)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def _synthesize_piper_cli(
    text: str, model_path: str,
    length_scale: float = 1.0, sentence_silence: float = 0.4,
) -> io.BytesIO:
    """Fallback: call the piper CLI binary with prosody flags."""
    import subprocess

    piper_bin = CONFIG["PIPER_BINARY"]
    log.info("Piper CLI: binary=%s, model=%s, length_scale=%.2f, sentence_silence=%.2f",
             piper_bin, model_path, length_scale, sentence_silence)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            piper_bin,
            "--model", model_path,
            "--output_file", tmp_path,
            "--length-scale", str(length_scale),
            "--sentence-silence", str(sentence_silence),
        ]
        proc = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        if proc.stderr:
            log.info("Piper CLI stderr: %s", proc.stderr[:500])

        buf = io.BytesIO(Path(tmp_path).read_bytes())
        return buf
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def synthesize_huggingface(text: str, model_name: str) -> io.BytesIO:
    """Synthesize text → MP3 via a HuggingFace TTS pipeline."""
    import numpy as np

    pipe = _get_hf_pipeline(model_name)
    log.info("HF TTS: model=%s, text_len=%d", model_name, len(text))

    # HF TTS pipelines may struggle with very long texts — chunk it
    chunks = chunk_text(text, 5000)
    combined = AudioSegment.empty()

    # 400ms silence between chunks for natural paragraph pauses
    paragraph_silence = AudioSegment.silent(duration=400)

    for i, chunk in enumerate(chunks):
        result = pipe(chunk)
        audio_array = np.array(result["audio"]).flatten()
        sampling_rate = result["sampling_rate"]

        pcm = (audio_array * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sampling_rate)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)
        combined += AudioSegment.from_wav(wav_buf)
        if i < len(chunks) - 1:
            combined += paragraph_silence
        log.info("  HF chunk %d/%d done", i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    project_dir = Path(__file__).parent
    # Discover .onnx models in models/ subfolder
    models_dir = project_dir / "models"
    onnx_models = sorted(str(p) for p in models_dir.glob("*.onnx")) if models_dir.is_dir() else []
    # Discover PDFs in input/ folder
    input_dir = project_dir / "input"
    input_pdfs = sorted(p.name for p in input_dir.glob("*.pdf")) if input_dir.is_dir() else []
    return render_template(
        "index.html",
        config=CONFIG,
        onnx_models=onnx_models,
        input_pdfs=input_pdfs,
    )


def _err(msg: str, status: int = 400):
    """Return a JSON error so the fetch-based frontend can display it."""
    return jsonify({"error": msg}), status


@app.route("/api/input-files")
def api_input_files():
    """Return list of PDFs in the input/ folder."""
    input_dir = Path(__file__).parent / "input"
    pdfs = sorted(p.name for p in input_dir.glob("*.pdf")) if input_dir.is_dir() else []
    return jsonify({"files": pdfs})


@app.route("/api/polly-voices")
def api_polly_voices():
    """Return available Polly voices grouped by engine from AWS."""
    try:
        import boto3
        polly = boto3.client("polly", region_name=CONFIG["AWS_REGION"])
        resp = polly.describe_voices()
        voices = []
        for v in resp.get("Voices", []):
            voices.append({
                "id": v["Id"],
                "name": v["Name"],
                "gender": v["Gender"],
                "language": v["LanguageName"],
                "language_code": v["LanguageCode"],
                "engines": v.get("SupportedEngines", []),
            })
        voices.sort(key=lambda v: (v["language"], v["name"]))
        return jsonify({"voices": voices})
    except Exception as e:
        log.warning("Failed to fetch Polly voices: %s", e)
        return jsonify({"error": str(e), "voices": []})


@app.route("/api/pdf-info", methods=["POST"])
def api_pdf_info():
    """Return page count (and per-page text-presence) for a PDF.

    Accepts either an uploaded file or an `input_file` name from input/.
    """
    input_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    if input_name:
        pdf_path = Path(__file__).parent / "input" / Path(input_name).name
        if not pdf_path.is_file():
            return _err(f"File not found in input/: {input_name}")
        reader = PdfReader(str(pdf_path))
    elif pdf_file:
        reader = PdfReader(pdf_file.stream)
        pdf_file.stream.seek(0)
    else:
        return _err("No PDF provided.")

    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").strip()
        pages.append({"page": i + 1, "has_text": len(txt) > 20, "chars": len(txt)})

    # --- Chapter / TOC detection ---
    chapters = []

    # 1) Try PDF outline (bookmarks)
    if reader.outline:
        def _walk_outline(items, depth=0):
            for item in items:
                if isinstance(item, list):
                    _walk_outline(item, depth + 1)
                else:
                    try:
                        pn = reader.get_destination_page_number(item)
                        chapters.append({
                            "title": item.title,
                            "page": pn + 1,
                            "depth": depth,
                        })
                    except Exception:
                        pass
        _walk_outline(reader.outline)

    # 2) Fallback: heuristic scan for chapter headings in page text
    if not chapters:
        chapter_re = re.compile(
            r'^(?:chapter|part|section|prologue|epilogue|introduction|conclusion)'
            r'(?:\s+[\dIVXLCivxlc]+)?',
            re.IGNORECASE,
        )
        for i, page in enumerate(reader.pages):
            txt = (page.extract_text() or "")[:300].strip()
            first_line = txt.split("\n")[0].strip() if txt else ""
            if first_line and chapter_re.match(first_line):
                chapters.append({"title": first_line, "page": i + 1, "depth": 0})

    return jsonify({"total_pages": total, "pages": pages, "chapters": chapters})


@app.route("/convert", methods=["POST"])
def convert():
    t0 = time.time()

    # --- Collect & validate form data ---
    save_to_output = request.form.get("save_to_output") == "1"
    input_file_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    # Determine PDF source: input/ folder file or uploaded file
    if input_file_name:
        pdf_path = Path(__file__).parent / "input" / Path(input_file_name).name
        if not pdf_path.is_file():
            return _err(f"File not found in input/: {input_file_name}")
        file_stream = open(pdf_path, "rb")
        pdf_filename = pdf_path.name
    elif pdf_file and pdf_file.filename:
        ext = Path(pdf_file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _err(f"Invalid file type '{ext}'. Only PDF files are accepted.")
        header = pdf_file.stream.read(4)
        pdf_file.stream.seek(0)
        if header[:4] != PDF_MAGIC:
            return _err("The uploaded file does not appear to be a valid PDF.")
        file_stream = pdf_file.stream
        pdf_filename = pdf_file.filename
    else:
        return _err("Please upload a PDF file or select one from the input folder.")

    page_range = request.form.get("page_range", "all").strip()
    backend = request.form.get("backend", CONFIG["DEFAULT_BACKEND"])
    voice = request.form.get("voice", "").strip()

    if backend not in ("polly", "piper", "huggingface"):
        return _err(f"Unknown backend: {backend}")

    log.info(
        "Request: backend=%s, pages=%s, voice=%s, file=%s, save_to_output=%s",
        backend,
        page_range,
        voice or "(default)",
        pdf_filename,
        save_to_output,
    )

    # --- Extract text ---
    try:
        text, label = extract_text_from_pdf(file_stream, page_range)
    except ValueError as e:
        return _err(str(e))
    finally:
        if input_file_name:
            file_stream.close()

    if not text.strip():
        return _err(
            "No extractable text found in the selected pages. "
            "The PDF may contain scanned images instead of text."
        )

    if len(text) > MAX_TEXT_CHARS:
        return _err(
            f"Extracted text is too long ({len(text):,} chars). "
            f"Please select fewer pages (limit: {MAX_TEXT_CHARS:,} chars)."
        )

    log.info("Extracted text: %d chars from %s", len(text), label)

    # --- Synthesize ---
    provider_detail = "default"
    polly_chars_billed: int | None = None
    polly_cost_usd: float | None = None
    try:
        if backend == "polly":
            voice_id = voice or CONFIG["POLLY_VOICE_ID"]
            engine = request.form.get("engine", CONFIG["POLLY_ENGINE"])
            provider_detail = f"{voice_id}_{engine}"
            mp3_buf, polly_chars_billed = synthesize_polly(text, voice_id, engine)
            rate = _POLLY_RATES.get(engine.lower(), _POLLY_RATES["neural"])
            polly_cost_usd = polly_chars_billed * rate / 1_000_000

        elif backend == "piper":
            model_path = voice or CONFIG["PIPER_MODEL"]
            provider_detail = Path(model_path).name
            # Read prosody overrides from the form (fall back to .env defaults)
            def _float_or_none(key):
                v = request.form.get(key, "").strip()
                return float(v) if v else None
            mp3_buf = synthesize_piper(
                text,
                model_path,
                length_scale=_float_or_none("length_scale"),
                noise_scale=_float_or_none("noise_scale"),
                noise_w_scale=_float_or_none("noise_w_scale"),
                sentence_silence=_float_or_none("sentence_silence"),
            )

        elif backend == "huggingface":
            model_name = voice or CONFIG["HF_MODEL"]
            provider_detail = model_name
            mp3_buf = synthesize_huggingface(text, model_name)

    except Exception as e:
        log.exception("TTS synthesis failed")
        return _err(f"TTS error ({backend}): {e}", 500)

    elapsed = time.time() - t0
    log.info("Done in %.1fs — serving MP3", elapsed)

    filename = build_output_filename(pdf_filename, label, backend, provider_detail)

    extra: dict = {}
    if polly_chars_billed is not None:
        extra["polly_chars_billed"] = polly_chars_billed
        extra["polly_cost_usd"] = round(polly_cost_usd, 6)
        log.info("Polly cost summary: %d chars → $%.4f", polly_chars_billed, polly_cost_usd)

    if save_to_output:
        output_path = Path(__file__).parent / "output" / filename
        output_path.write_bytes(mp3_buf.read())
        mp3_buf.seek(0)
        log.info("Saved to %s", output_path)
        return jsonify({"saved": True, "filename": filename, "path": str(output_path), **extra})

    return send_file(
        mp3_buf,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=filename,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, debug=True)
