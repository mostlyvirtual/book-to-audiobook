"""
Book-to-Audiobook Flask App
============================
Upload a PDF or EPUB, pick pages and a TTS backend, download an MP3.

Backends:
  - polly   : Amazon Polly (requires AWS credentials)
  - piper   : Local Piper TTS (piper-tts Python lib → CLI fallback)
  - huggingface : HF Transformers TTS pipeline

All configuration lives in .env — see .env for docs.
"""

import io
import json
import os
import re
import time
import uuid
import wave
import logging
import tempfile
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env before anything reads os.getenv

# ---------------------------------------------------------------------------
# Force all model caches into a project-local .cache/ directory
# Must run BEFORE importing any HF / Supertonic libraries.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
_PROJECT_CACHE = _PROJECT_ROOT / ".cache"
_PROJECT_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(_PROJECT_CACHE / "huggingface"))
os.environ.setdefault("SUPERTONIC_CACHE_DIR", str(_PROJECT_CACHE / "supertonic"))

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
import fitz  # PyMuPDF
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Configuration — all values come from .env (see .env for docs)
# ---------------------------------------------------------------------------

CONFIG = {
    # General
    "DEFAULT_BACKEND": os.getenv("TTS_BACKEND", "kokoro"),
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
    # Kokoro
    "KOKORO_VOICE": os.getenv("KOKORO_VOICE", "am_michael"),
    "KOKORO_SPEED": float(os.getenv("KOKORO_SPEED", "1.0")),
    "KOKORO_LANG": os.getenv("KOKORO_LANG", "a"),
    # Supertonic
    "SUPERTONIC_VOICE": os.getenv("SUPERTONIC_VOICE", "F1"),
    "SUPERTONIC_LANG": os.getenv("SUPERTONIC_LANG", "en"),
    "SUPERTONIC_SPEED": float(os.getenv("SUPERTONIC_SPEED", "1.05")),
    "SUPERTONIC_SILENCE": float(os.getenv("SUPERTONIC_SILENCE", "0.3")),
}

MAX_UPLOAD_MB = 50
MAX_TEXT_CHARS = 500_000  # safety cap — ~200 pages of text
POLLY_CHUNK_CHARS = 2800  # safe limit per Polly SynthesizeSpeech call
ALLOWED_EXTENSIONS = {".pdf", ".epub"}
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
# Job progress tracking (polled by frontend during conversion)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}  # job_id -> progress + control state
_jobs_lock = threading.Lock()
_job_context = threading.local()  # per-request job_id for lazy loaders


class JobCancelledError(Exception):
    """Raised inside a synthesis loop when the user cancels the job."""


def _make_on_progress(job_id: str | None, phase: str = "synthesizing"):
    """Return an on_progress(current, total) callback that:
    - raises JobCancelledError if the job was cancelled
    - blocks until resumed if the job is paused
    - reports progress to the jobs tracker
    """
    def _cb(current: int, total: int) -> None:
        if _check_cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} cancelled")
        _wait_if_paused(job_id)
        _report_progress(job_id, current, total, phase=phase)
    return _cb


def _report_downloading(label: str) -> None:
    """Used by lazy loaders to signal a first-run model download to the frontend."""
    job_id = getattr(_job_context, 'job_id', None)
    _report_progress(job_id, 0, 0, phase="downloading")
    log.info("Downloading model: %s (this only happens once)", label)


def _report_progress(job_id: str | None, current: int, total: int, phase: str = "synth"):
    """Update progress for a running job (thread-safe)."""
    if not job_id:
        return
    with _jobs_lock:
        if job_id not in _jobs:
            e = threading.Event()
            e.set()  # running by default
            _jobs[job_id] = {"cancel_requested": False, "pause_event": e}
        _jobs[job_id].update({"phase": phase, "current": current, "total": total})


def _clear_job(job_id: str | None):
    """Remove a finished job from the tracker."""
    if not job_id:
        return
    with _jobs_lock:
        _jobs.pop(job_id, None)


def _request_cancel(job_id: str) -> None:
    """Request cancellation of a running job."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["cancel_requested"] = True
            _jobs[job_id]["pause_event"].set()  # wake paused jobs so they notice cancel


def _request_pause(job_id: str) -> None:
    """Pause a running job between chapters."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["pause_event"].clear()


def _request_resume(job_id: str) -> None:
    """Resume a paused job."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["pause_event"].set()


def _check_cancelled(job_id: str | None) -> bool:
    """Return True if the job has been cancelled."""
    if not job_id:
        return False
    with _jobs_lock:
        info = _jobs.get(job_id)
        return bool(info and info.get("cancel_requested"))


def _wait_if_paused(job_id: str | None) -> None:
    """Block until the job is resumed (or cancelled)."""
    if not job_id:
        return
    while True:
        with _jobs_lock:
            info = _jobs.get(job_id)
            if info is None or info.get("cancel_requested"):
                return
            event = info["pause_event"]
        if event.wait(timeout=0.5):
            return


# ---------------------------------------------------------------------------
# Lazy-loaded singletons (heavy models loaded once)
# ---------------------------------------------------------------------------

_piper_voice = None
_hf_pipelines: dict[str, object] = {}
_kokoro_pipelines: dict[str, object] = {}
_supertonic_tts = None
_xtts_models: dict[str, object] = {}
_speecht5_cache: dict[str, object] = {}
_piper_cli_noise_flags_supported: bool | None = None
_polly_status_cache = {
    "checked_at": 0.0,
    "result": {"ready": False, "note": "Checking AWS credentials..."},
}
_POLLY_STATUS_TTL_SECONDS = 60


def _validate_piper_model_files(model_path: str) -> str:
    """Validate Piper model + sidecar config JSON and return config path.

    Raises ValueError with actionable message when files are missing/corrupt.
    """
    model = Path(model_path)
    if not model.is_file():
        raise ValueError(f"Piper model not found: {model}")

    config_path = Path(f"{model_path}.json")
    if not config_path.is_file():
        raise ValueError(
            f"Piper config JSON not found: {config_path}. "
            "Ensure the model sidecar .json file exists."
        )

    try:
        with config_path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("config JSON is empty or invalid object")
    except Exception as e:
        raise ValueError(
            f"Piper config JSON is invalid: {config_path} ({e}). "
            "Redownload this voice model or select a different one."
        )

    return str(config_path)


def _get_piper_voice(model_path: str):
    """Load PiperVoice once and cache it."""
    global _piper_voice
    if _piper_voice is None or _piper_voice._model_path != model_path:
        try:
            from piper import PiperVoice
        except ImportError:
            log.error("Piper not installed. Install with: uv sync --extra piper")
            raise
        log.info("Loading Piper model: %s", model_path)
        _piper_voice = PiperVoice.load(model_path)
        _piper_voice._model_path = model_path  # tag for cache check
    return _piper_voice


def _get_hf_pipeline(model_name: str):
    """Load HF TTS pipeline once per model and cache it."""
    global _hf_pipelines
    if model_name not in _hf_pipelines:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            log.error("Transformers not installed. Install with: uv sync --extra huggingface")
            raise
        _report_downloading(f"HuggingFace TTS: {model_name}")
        log.info("Loading HF TTS model: %s", model_name)
        _hf_pipelines[model_name] = hf_pipeline("text-to-speech", model=model_name)
    return _hf_pipelines[model_name]


def _get_kokoro_pipeline(lang_code: str):
    """Load Kokoro TTS pipeline once and cache it.
    
    Args:
        lang_code: Language code (e.g., 'a' for American English, 'b' for British)
    Returns:
        KPipeline instance
    """
    global _kokoro_pipelines
    if lang_code not in _kokoro_pipelines:
        try:
            from kokoro import KPipeline
            _report_downloading(f"Kokoro (lang={lang_code})")
            log.info("Loading Kokoro TTS pipeline for lang=%s", lang_code)
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
        except ImportError:
            log.error("Kokoro not installed. Install with: uv sync --dev")
            raise
    return _kokoro_pipelines[lang_code]


def _get_supertonic_tts():
    """Load Supertonic TTS model once and cache it.
    
    Note: First call will download ~305MB model on demand.
    """
    global _supertonic_tts
    if _supertonic_tts is None:
        try:
            from supertonic import TTS
            _report_downloading("Supertonic (~305 MB)")
            log.info("Loading Supertonic TTS (may download ~305MB model on first run)")
            _supertonic_tts = TTS(auto_download=True)
        except ImportError:
            log.error("Supertonic not installed. Install with: uv sync --extra supertonic")
            raise
    return _supertonic_tts


# ---------------------------------------------------------------------------
# Text preprocessing — makes TTS output sound like an audiobook
# ---------------------------------------------------------------------------

_easyocr_reader = None


def _get_easyocr():
    """Lazy-load EasyOCR reader (only for scanned PDFs)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        log.info("Loading EasyOCR (first scanned page)")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


def _join_wrapped_lines(text: str) -> str:
    """Join lines where current ends without punctuation and next starts lowercase.
    
    Fixes document hard-wraps like:
      "This is impor-
       tant information" → "This is important information"
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        current = lines[i].rstrip()
        # Check if next line exists and current line ends with a dash
        if i + 1 < len(lines) and current.endswith('-'):
            next_line = lines[i + 1].lstrip()
            # If next line starts with lowercase letter, it's a continuation
            if next_line and next_line[0].islower():
                result.append(current[:-1] + next_line)
                i += 2
                continue
        result.append(current)
        i += 1
    return '\n'.join(result)


def _expand_abbreviations(text: str) -> str:
    """Expand common abbreviations to help TTS pronunciation.
    
    Converts: Mr., Dr., e.g., etc., i.e., and ~15 more common abbreviations.
    Uses word-boundary matching to avoid partial hits.
    """
    abbreviations = {
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Misses',
        r'\bMs\.': 'Ms',
        r'\bDr\.': 'Doctor',
        r'\bProf\.': 'Professor',
        r'\bSt\.': 'Saint',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\betc\.': 'et cetera',
        r'\bvs\.': 'versus',
        r'\bfig\.': 'figure',
        r'\bno\.': 'number',
        r'\bvol\.': 'volume',
        r'\bed\.': 'edition',
        r'\bpgs?\.': 'page',
        r'\bet al\.': 'and others',
    }
    
    for abbrev, expansion in abbreviations.items():
        text = re.sub(abbrev, expansion, text, flags=re.IGNORECASE)
    
    return text


def _convert_numbers(text: str) -> str:
    """Convert numbers to words for better TTS pronunciation.
    
    Handles: integers, decimals, ordinals (1st→first), years (1800-2099),
    currency ($123.45), percentages (45%). Caps at 6-digit numbers.
    """
    from num2words import num2words
    
    # Ordinals: 1st, 2nd, 3rd, 21st, etc.
    def _ordinal_replace(match):
        num_str = match.group(1)
        try:
            num = int(num_str)
            if 1 <= num <= 999999:
                return num2words(num, ordinal=True, lang='en')
        except (ValueError, TypeError):
            pass
        return match.group(0)
    
    text = re.sub(r'\b(\d{1,6})(?:st|nd|rd|th)\b', _ordinal_replace, text, flags=re.IGNORECASE)
    
    # Currency: $123.45 or £5.99
    def _currency_replace(match):
        symbol = match.group(1)
        amount_str = match.group(2).replace(',', '')
        try:
            amount = float(amount_str)
            if amount < 1000000:
                words = num2words(amount, to='currency', lang='en')
                return words
        except (ValueError, TypeError):
            pass
        return match.group(0)
    
    text = re.sub(r'([$£€])(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', _currency_replace, text)
    
    # Percentages: 45% → forty-five percent
    def _percent_replace(match):
        num_str = match.group(1)
        try:
            num = float(num_str)
            if num < 1000000:
                words = num2words(num, lang='en')
                return f"{words} percent"
        except (ValueError, TypeError):
            pass
        return match.group(0)
    
    text = re.sub(r'\b(\d{1,3}(?:\.\d{1,2})?)\s*%', _percent_replace, text)
    
    # Years (1800-2099): recognize patterns like "in 1995" or "from 2020"
    # This is conservative — only replace if surrounded by year-context words
    def _year_replace(match):
        num_str = match.group(1)
        try:
            num = int(num_str)
            if 1800 <= num <= 2099:
                if num % 100 == 0:
                    # Round century: 1900 → nineteen hundred
                    return num2words(num, lang='en')
                else:
                    # Split: 1995 → nineteen ninety-five
                    hundreds = num // 100
                    remainder = num % 100
                    h_words = num2words(hundreds, lang='en')
                    if remainder == 0:
                        return h_words + ' hundred'
                    else:
                        r_words = num2words(remainder, lang='en')
                        return f"{h_words} {r_words}"
        except (ValueError, TypeError):
            pass
        return match.group(0)
    
    # Only expand years in year-like contexts
    text = re.sub(r'\b(1[89]\d{2}|20\d{2})\b(?=\s+(?:was|were|in|from|to|until|began|ended|through))', _year_replace, text, flags=re.IGNORECASE)
    
    return text


def _remove_artifacts(text: str) -> str:
    """Remove extraction artifacts: citations, page numbers, URLs.
    
    Strips citation-like brackets, page numbers, and URLs while preserving
    normal bracketed prose.
    """
    # Remove citation brackets: [1], [1, 2], [1-3], [Ref 45], [Refs 2, 4]
    citation_patterns = (
        r"\[(?:\d+\s*(?:[,;-]\s*\d+\s*)*)\]",
        r"\[(?:ref|refs|reference|references)\s+\d+(?:\s*(?:[,;-]\s*\d+\s*)*)\]",
    )
    for pattern in citation_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove footer/header page numbers (lines that are just digits)
    text = re.sub(r'^[\s\-\d]+$', '', text, flags=re.MULTILINE)
    
    return text


# ---------------------------------------------------------------------------
# PDF running-header / footer stripping
# ---------------------------------------------------------------------------

def _build_pdf_header_patterns(doc) -> dict:
    """Scan a PDF to detect running headers via frequency analysis.

    Accepts a fitz Document object.
    Returns a dict with:
      - header_lines: set of exact first-line strings (Strategy 1: exact match)
      - header_prefix: common prefix string (Strategy 2: shared prefix)
      - continuation_lines: set of short second-line fragments that are header overflow
    Only one of header_lines/header_prefix will be populated.
    Returns {} if no running headers detected.
    """
    from collections import Counter
    import os

    total_pages = len(doc)
    if total_pages < 4:
        return {}

    first_lines: list[str] = []
    second_line_counter: Counter[str] = Counter()

    for pg in range(total_pages):
        text = doc.load_page(pg).get_text() or ""
        lines = text.split("\n")
        if not lines:
            first_lines.append("")
            continue
        first_lines.append(lines[0].strip())
        if len(lines) >= 2:
            second_line_counter[lines[1].strip()] += 1

    exact_counter = Counter(first_lines)

    # Strategy 1: Exact match — first lines appearing on ≥ 25% of pages
    threshold = max(3, total_pages * 0.25)
    header_lines = {
        line for line, count in exact_counter.items()
        if count >= threshold and line
    }

    # Strategy 2: Common prefix — if exact match fails, look for a shared prefix
    # (handles PDFs where the header changes per section, e.g. "Title | Ch. N")
    header_prefix = ""
    if not header_lines:
        frequent = [
            line for line, count in exact_counter.items()
            if count >= 2 and line
        ]
        if len(frequent) >= 2:
            prefix = os.path.commonprefix(frequent)
            prefix_matches = sum(1 for l in first_lines if l.startswith(prefix))
            if len(prefix) >= 10 and prefix_matches >= total_pages * 0.5:
                header_prefix = prefix

    if not header_lines and not header_prefix:
        return {}

    # Header continuations: short second lines that repeat on ≥ 3 pages
    continuation_lines: set[str] = set()
    for line, count in second_line_counter.items():
        if count >= 3 and len(line) < 25:
            if not re.match(r"^(CHAPTER\s+\d|PROLOGUE|EPILOGUE)", line, re.I):
                continuation_lines.add(line)

    result: dict = {"continuation_lines": continuation_lines}
    if header_lines:
        result["header_lines"] = header_lines
    if header_prefix:
        result["header_prefix"] = header_prefix
    return result


_CHAPTER_LABEL_RE = re.compile(
    r"^(CHAPTER\s+\d+|PROLOGUE|EPILOGUE|BIBLIOGRAPHY"
    r"|INTRODUCTION|FOREWORD|PREFACE|AFTERWORD"
    r"|APPENDIX(?:\s+[A-Z])?)$",
    re.IGNORECASE,
)


def _strip_page_headers(page_text: str, patterns: dict) -> str:
    """Remove running headers, continuations, and chapter-label echoes
    from a single page's extracted text.

    Safe: only removes lines at the TOP of the page that match detected
    patterns — never touches mid-page content.
    """
    if not patterns:
        return page_text

    header_lines = patterns.get("header_lines", set())
    header_prefix = patterns.get("header_prefix", "")
    continuation_lines = patterns.get("continuation_lines", set())

    lines = page_text.split("\n")
    drop = 0  # number of leading lines to remove

    if not lines:
        return page_text

    # Step 1: Strip running header (first line)
    first = lines[0].strip()
    is_header = (first in header_lines) or (header_prefix and first.startswith(header_prefix))
    if is_header:
        drop = 1

        # Step 2: Strip header continuation (second line, if present)
        if len(lines) > drop and lines[drop].strip() in continuation_lines:
            drop += 1

        # Step 3: Strip chapter-label echo on chapter-opening pages.
        # Pattern: "CHAPTER N" + "Subtitle" + "CHAPTER N Subtitle body text..."
        # The third line already contains the label + subtitle merged into body,
        # so the standalone label and subtitle lines are decorative duplicates.
        remaining = lines[drop:]
        if len(remaining) >= 3:
            l1 = remaining[0].strip()
            l2 = remaining[1].strip()
            l3 = remaining[2].strip()
            if _CHAPTER_LABEL_RE.match(l1):
                # Check if l3 starts with "LABEL SUBTITLE" (the echo)
                expected_prefix = l1 + " " + l2
                # Normalize whitespace for comparison
                l3_norm = re.sub(r"\s+", " ", l3)
                expected_norm = re.sub(r"\s+", " ", expected_prefix)
                if l3_norm.startswith(expected_norm):
                    drop += 2  # skip the standalone label + subtitle

    return "\n".join(lines[drop:])


def preprocess_text_for_speech(text: str) -> str:
    """Clean up extracted text for more natural TTS output.

    Pipeline:
    0. NFKC Unicode normalization
    1. Join hard-wrapped lines (document line breaks)
    2. Expand abbreviations (Mr. → Mister)
    3. Convert numbers (45% → forty-five percent)
    4. Normalize ligatures, whitespace, punctuation
    5. Remove artifacts (citations, URLs, page numbers)
    6. Add sentence-ending punctuation & pause markers
    """
    # Step 0: NFKC normalization (resolves ligatures, full-width chars, etc.)
    text = unicodedata.normalize('NFKC', text)

    # Step 1: Join wrapped lines
    text = _join_wrapped_lines(text)
    
    # Step 2: Expand abbreviations
    text = _expand_abbreviations(text)
    
    # Step 3: Convert numbers
    text = _convert_numbers(text)
    
    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize common extraction artifacts (ligatures)
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

    # Single newlines mid-paragraph -> space (document line wrapping)
    text = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", text)

    # Clean up doubled periods from above transforms
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\.\s*\.", ".", text)
    
    # Step 4: Remove artifacts (citations, URLs, page numbers)
    text = _remove_artifacts(text)

    # Ensure text ends with punctuation
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    return text


# ---------------------------------------------------------------------------
# Document helpers (PDF + EPUB)
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


def _is_scanned_page(page) -> bool:
    """Return True if a fitz page appears to be a scanned image (no selectable text)."""
    return len(page.get_text().strip()) < 50 and len(page.get_images()) > 0


def _ocr_page(page, dpi: int = 300) -> str:
    """Run EasyOCR on a fitz page and return joined text."""
    import numpy as np
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:  # RGBA → RGB
        img = img[:, :, :3]
    results = _get_easyocr().readtext(img, detail=0)
    return "\n".join(results)


def _extract_page_text_fitz(page, header_threshold: int = 50, footer_threshold: int = 50) -> str:
    """Extract text from a fitz page, filtering header/footer zones by y-coordinate.

    Blocks whose bottom edge (y1) is above *header_threshold* pixels from the top,
    or whose top edge (y0) is more than *page_height - footer_threshold* pixels down,
    are discarded.  Falls back to OCR for scanned pages.
    """
    ph = page.rect.height
    lines: list[str] = []
    for block in page.get_text("blocks"):
        # block = (x0, y0, x1, y1, text, block_no, block_type)
        x0, y0, x1, y1, text, *_ = block
        if y1 < header_threshold or y0 > ph - footer_threshold:
            continue
        stripped = text.strip()
        if stripped:
            lines.append(stripped)
    result = "\n".join(lines)
    # OCR fallback for scanned pages
    if len(result.strip()) < 50 and _is_scanned_page(page):
        log.info("Scanned page detected — running OCR")
        result = _ocr_page(page)
    return result


def extract_text_from_pdf(file_stream, page_range_str: str) -> tuple[str, str]:
    """Extract and preprocess text from selected PDF pages."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    total = len(doc)
    indices = parse_page_range(page_range_str, total)

    # Detect running headers across the whole document
    header_patterns = _build_pdf_header_patterns(doc)

    parts: list[str] = []
    for i in indices:
        text = _extract_page_text_fitz(doc.load_page(i))
        text = _strip_page_headers(text, header_patterns)
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


def _load_epub_entries(file_stream) -> list[dict]:
    """Return EPUB spine entries with extracted text and display titles."""
    import zipfile
    from xml.etree import ElementTree as ET
    from bs4 import BeautifulSoup

    if hasattr(file_stream, "seek"):
        file_stream.seek(0)

    with zipfile.ZipFile(file_stream) as zf:
        try:
            container_xml = zf.read("META-INF/container.xml")
            root = ET.fromstring(container_xml)
            rootfile = root.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
            if rootfile is None:
                raise ValueError("Invalid EPUB: no rootfile found")
            opf_path = rootfile.get("full-path")
            if not opf_path:
                raise ValueError("Invalid EPUB: rootfile has no full-path")
        except (KeyError, ET.ParseError) as e:
            raise ValueError(f"Invalid EPUB: could not parse container: {e}")

        try:
            opf_xml = zf.read(opf_path)
            opf_root = ET.fromstring(opf_xml)
        except (KeyError, ET.ParseError) as e:
            raise ValueError(f"Invalid EPUB: could not parse OPF: {e}")

        ns = {
            "opf": "http://www.idpf.org/2007/opf",
            "dc": "http://purl.org/dc/elements/1.1/",
        }
        manifest: dict[str, dict] = {}
        for item in opf_root.findall(".//opf:item", ns):
            item_id = item.get("id")
            href = item.get("href")
            if item_id and href:
                manifest[item_id] = {"href": href}

        spine_ids = []
        for itemref in opf_root.findall(".//opf:itemref", ns):
            item_id = itemref.get("idref")
            if item_id in manifest:
                spine_ids.append(item_id)

        if not spine_ids:
            raise ValueError("EPUB has no content spine")

        metadata_title = opf_root.findtext(".//dc:title", default="", namespaces=ns).strip()
        opf_dir = Path(opf_path).parent
        entries: list[dict] = []

        for idx, item_id in enumerate(spine_ids, start=1):
            full_path = (opf_dir / manifest[item_id]["href"]).as_posix()
            title = f"Chapter {idx}"
            text = ""
            try:
                content = zf.read(full_path)
                soup = BeautifulSoup(content, "html.parser")
                for el in soup(["script", "style"]):
                    el.decompose()

                for tag in soup.find_all(["h1", "h2", "h3", "title"]):
                    candidate = tag.get_text(" ", strip=True)
                    if candidate:
                        title = candidate
                        break

                text = soup.get_text(separator="\n").strip()
                if metadata_title and title == metadata_title:
                    title = f"Chapter {idx}"
            except KeyError:
                log.warning("EPUB: item not found at %s", full_path)

            entries.append({
                "page": idx,
                "title": title,
                "text": text,
                "chars": len(text),
                "has_text": len(text) > 20,
            })

        return entries


def extract_text_from_epub(file_stream, page_range_str: str) -> tuple[str, str]:
    """Extract and preprocess text from selected EPUB chapters.
    
    For EPUB, "pages" are actually spine items (chapters/sections).
    Supports page_range like "all", "1-5", "3".
    """
    try:
        entries = _load_epub_entries(file_stream)
        indices = parse_page_range(page_range_str, len(entries))
        parts = [entries[idx]["text"] for idx in indices if entries[idx]["text"].strip()]
        if not parts:
            raise ValueError("No readable content found in EPUB")

        raw = "\n".join(parts)
        cleaned = preprocess_text_for_speech(raw)

        if page_range_str.strip().lower() == "all":
            label = "all"
        elif len(indices) == 1:
            label = f"ch{indices[0]+1}"
        else:
            label = f"ch{indices[0]+1}-{indices[-1]+1}"

        return cleaned, label
    except Exception as e:
        log.exception("EPUB extraction failed")
        raise ValueError(f"Failed to extract EPUB: {e}")


def _slug_token(value: str, max_len: int = 50) -> str:
    """Convert text to a compact filesystem-safe token."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    token = re.sub(r"-+", "-", token).strip("-._")
    return (token[:max_len] or "na")


def build_output_filename(
    source_filename: str,
    page_label: str,
    backend: str,
    provider_detail: str,
    prosody: dict | None = None,
) -> str:
    """Create stable output names:

    <ISO-UTC>_<book>_<pages>_<backend>_<voice-or-model>[_prosody].mp3
    """
    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    book = _slug_token(Path(source_filename).stem, max_len=60)
    pages = _slug_token(page_label, max_len=24)
    backend_token = _slug_token(backend, max_len=16)
    detail = _slug_token(provider_detail, max_len=80)
    base = f"{iso_ts}_{book}_{pages}_{backend_token}_{detail}"
    if prosody:
        tags = "_".join(f"{k}{v}" for k, v in sorted(prosody.items()) if v is not None)
        if tags:
            base += f"_{_slug_token(tags, max_len=80)}"
    return f"{base}.mp3"


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


def synthesize_polly(text: str, voice_id: str, engine: str, on_progress=None) -> tuple[io.BytesIO, int]:
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
        if on_progress:
            on_progress(i + 1, len(chunks))

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
    on_progress=None,
) -> io.BytesIO:
    """Synthesize text → MP3 via Piper TTS with prosody controls.

    Uses the Python API's synthesize() iterator so we can inject silence
    between sentences for natural audiobook pacing.
    """
    ls = length_scale if length_scale is not None else CONFIG["PIPER_LENGTH_SCALE"]
    ns = noise_scale if noise_scale is not None else CONFIG["PIPER_NOISE_SCALE"]
    nws = noise_w_scale if noise_w_scale is not None else CONFIG["PIPER_NOISE_W_SCALE"]
    ss = sentence_silence if sentence_silence is not None else CONFIG["PIPER_SENTENCE_SILENCE"]

    config_path = _validate_piper_model_files(model_path)

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
                if on_progress:
                    on_progress(i + 1, i + 1)  # total unknown, keep current == total

    except Exception as e:
        log.warning("Piper Python API failed (%s), falling back to CLI", e)
        wav_buf = _synthesize_piper_cli(
            text,
            model_path,
            config_path,
            length_scale=ls,
            sentence_silence=ss,
            noise_scale=ns,
            noise_w_scale=nws,
        )

    wav_buf.seek(0)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def _synthesize_piper_cli(
    text: str, model_path: str, config_path: str,
    length_scale: float = 1.0,
    sentence_silence: float = 0.4,
    noise_scale: float = 0.667,
    noise_w_scale: float = 0.8,
) -> io.BytesIO:
    """Fallback: call the piper CLI binary with prosody flags."""
    import subprocess

    piper_bin = CONFIG["PIPER_BINARY"]
    log.info(
        "Piper CLI: binary=%s, model=%s, length_scale=%.2f, sentence_silence=%.2f, noise_scale=%.3f, noise_w_scale=%.3f",
        piper_bin, model_path, length_scale, sentence_silence, noise_scale, noise_w_scale,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not _piper_cli_supports_noise_flags(piper_bin):
            raise ValueError(
                "The installed Piper CLI does not support --noise-scale/--noise-w. "
                "Use the Python API path or upgrade Piper CLI."
            )
        cmd = [
            piper_bin,
            "--model", model_path,
            "--config", config_path,
            "--output_file", tmp_path,
            "--length-scale", str(length_scale),
            "--noise-scale", str(noise_scale),
            "--noise-w", str(noise_w_scale),
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


def _piper_cli_supports_noise_flags(piper_bin: str) -> bool:
    """Detect whether the installed Piper CLI supports noise tuning flags."""
    global _piper_cli_noise_flags_supported
    if _piper_cli_noise_flags_supported is not None:
        return _piper_cli_noise_flags_supported

    import subprocess

    try:
        proc = subprocess.run(
            [piper_bin, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Piper CLI binary not found: {piper_bin}") from e
    except Exception as e:
        raise ValueError(f"Failed to inspect Piper CLI flags: {e}") from e

    help_text = f"{proc.stdout}\n{proc.stderr}"
    _piper_cli_noise_flags_supported = (
        "--noise-scale" in help_text and ("--noise-w" in help_text or "--noise-w-scale" in help_text)
    )
    return _piper_cli_noise_flags_supported


def synthesize_huggingface(text: str, model_name: str, on_progress=None) -> io.BytesIO:
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
        if on_progress:
            on_progress(i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def synthesize_kokoro(text: str, voice: str, speed: float, lang_code: str, on_progress=None) -> io.BytesIO:
    """Synthesize text → MP3 via Kokoro TTS.
    
    Args:
        text: Text to synthesize
        voice: Voice name (e.g., 'af_bella', 'bf_emma')
        speed: Speech speed (0.5-2.0)
        lang_code: Language code (e.g., 'a' for American, 'b' for British)
    
    Returns:
        MP3 BytesIO buffer
    
    Note: Kokoro requires `espeak-ng` to be installed system-wide:
          macOS: `brew install espeak-ng`
          Linux: `apt-get install espeak-ng`
    """
    import numpy as np
    
    pipe = _get_kokoro_pipeline(lang_code)
    log.info("Kokoro TTS: voice=%s, speed=%.2f, lang=%s, text_len=%d", 
             voice, speed, lang_code, len(text))
    
    # Chunk text to avoid memory issues
    chunks = chunk_text(text, 2000)
    combined = AudioSegment.empty()
    
    # 400ms silence between chunks
    silence = AudioSegment.silent(duration=400)
    
    for i, chunk in enumerate(chunks):
        try:
            # Kokoro loads voice .pt files from HuggingFace on first use.
            # Temporarily allow network access even when HF_HUB_OFFLINE=1 so
            # voice packs can be downloaded and cached the first time.
            _prev_offline = os.environ.pop("HF_HUB_OFFLINE", None)
            try:
                # KPipeline.__call__ is a generator yielding Result objects
                chunk_samples = []
                for result in pipe(chunk, voice=voice, speed=speed):
                    if result.audio is not None:
                        chunk_samples.append(result.audio.cpu().numpy())
            finally:
                if _prev_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = _prev_offline

            if not chunk_samples:
                log.warning("  Kokoro chunk %d produced no audio", i + 1)
                continue

            samples = np.concatenate(chunk_samples)
            sample_rate = 24000  # Kokoro-82M outputs 24kHz audio
            
            # Normalize to 95% of int16 max to avoid clipping
            max_val = np.abs(samples).max()
            if max_val > 0:
                samples = samples * (0.95 / max_val)
            
            # Convert to int16
            pcm = (samples * 32767).astype(np.int16)
            
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            
            wav_buf.seek(0)
            segment = AudioSegment.from_wav(wav_buf)
            combined += segment
            
            if i < len(chunks) - 1:
                combined += silence
            
            log.info("  Kokoro chunk %d/%d done (%d chars)", i + 1, len(chunks), len(chunk))
            if on_progress:
                on_progress(i + 1, len(chunks))
        except Exception as e:
            log.exception("Kokoro chunk synthesis failed")
            raise
    
    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def synthesize_supertonic(
    text: str,
    voice_name: str,
    lang: str,
    speed: float = 1.05,
    silence_duration: float = 0.3,
    on_progress=None,
) -> io.BytesIO:
    """Synthesize text → MP3 via Supertonic TTS.
    
    Args:
        text: Text to synthesize
        voice_name: Voice identifier from TTS.voices
        lang: Language code (e.g., 'en')
        speed: Speech speed multiplier (default: 1.05)
        silence_duration: Silence between chunks in seconds (default: 0.3)
    
    Returns:
        MP3 BytesIO buffer
    
    Note: First call downloads ~305MB model. Uses pure ONNX, no torch required.
    """
    import numpy as np
    
    tts = _get_supertonic_tts()
    log.info("Supertonic TTS: voice=%s, lang=%s, speed=%.2f, silence=%.2f, text_len=%d",
             voice_name, lang, speed, silence_duration, len(text))
    
    # Chunk text
    chunks = chunk_text(text, 3000)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=max(0, int(silence_duration * 1000)))
    
    # Resolve configured style once; fallback to first available style if missing.
    style_names = list(getattr(tts, "voice_style_names", []) or [])
    style_name = voice_name if voice_name in style_names else (style_names[0] if style_names else "F1")
    voice_style = tts.get_voice_style(style_name)

    for i, chunk in enumerate(chunks):
        try:
            # Supertonic returns (audio, metadata); sample rate is on engine.
            synth_out = tts.synthesize(
                chunk, voice_style=voice_style, lang=lang,
                speed=speed, silence_duration=silence_duration, verbose=False,
            )

            if isinstance(synth_out, tuple):
                samples = synth_out[0]
            else:
                samples = synth_out

            sample_rate = int(getattr(tts, "sample_rate", 44100))

            # Some versions return shape (1, N)
            samples = np.array(samples).squeeze()
            
            # Ensure samples are correct dtype
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            # Normalize
            max_val = np.abs(samples).max()
            if max_val > 0:
                samples = samples * (0.95 / max_val)
            
            # Convert to int16
            pcm = (samples * 32767).astype(np.int16)
            
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
            
            wav_buf.seek(0)
            segment = AudioSegment.from_wav(wav_buf)
            combined += segment
            
            if i < len(chunks) - 1:
                combined += silence
            
            log.info("  Supertonic chunk %d/%d done (%d chars)", i + 1, len(chunks), len(chunk))
            if on_progress:
                on_progress(i + 1, len(chunks))
        except Exception as e:
            log.exception("Supertonic chunk synthesis failed")
            raise
    
    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


# ---------------------------------------------------------------------------
# XTTS-v2 — voice cloning (English + multilingual)
# ---------------------------------------------------------------------------

_CEDILLA_TO_COMMA = str.maketrans({
    "\u015f": "\u0219",  # ş → ș
    "\u0163": "\u021b",  # ţ → ț
    "\u015e": "\u0218",  # Ş → Ș
    "\u0162": "\u021a",  # Ţ → Ț
})


def _normalize_romanian(text: str) -> str:
    """Convert cedilla diacritics to comma-below (standard Romanian)."""
    return text.translate(_CEDILLA_TO_COMMA)


def _get_xtts_model(model_key: str = "base"):
    """Load XTTS-v2 model via Coqui TTS lib and cache it."""
    global _xtts_models
    if model_key not in _xtts_models:
        try:
            from TTS.api import TTS as CoquiTTS
            if model_key == "base":
                _report_downloading("XTTS-v2 base (~1.8 GB)")
                log.info("Loading XTTS-v2 base model (~1.8GB on first run)")
                _xtts_models[model_key] = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            else:
                _report_downloading(f"XTTS model: {model_key}")
                log.info("Loading XTTS model: %s", model_key)
                _xtts_models[model_key] = CoquiTTS(model_key)
        except ImportError:
            log.error("Coqui TTS not installed. Install with: uv sync --extra xtts")
            raise
    return _xtts_models[model_key]


def _get_xtts_ro_model():
    """Load the Romanian fine-tuned XTTS-v2 model and cache it."""
    global _xtts_models
    key = "xtts_ro"
    if key not in _xtts_models:
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            from huggingface_hub import snapshot_download

            _report_downloading("XTTS-v2 Romanian fine-tune (~1.8 GB)")
            log.info("Loading XTTS-v2 Romanian fine-tune (~1.8GB on first run)")
            model_dir = snapshot_download("eduardem/xtts-v2-romanian")
            config = XttsConfig()
            config.load_json(str(Path(model_dir) / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=False)
            _xtts_models[key] = model
        except ImportError:
            log.error("Coqui TTS not installed. Install with: uv sync --extra xtts")
            raise
    return _xtts_models[key]


def synthesize_xtts(
    text: str,
    language: str,
    reference_audio_path: str,
    speed: float = 1.0,
    on_progress=None,
) -> io.BytesIO:
    """Synthesize text → MP3 via XTTS-v2 with voice cloning.

    Uses the low-level inference API so that speaker conditioning latents
    are computed once and reused across all chunks (saves ~1-2s per chunk).
    """
    import numpy as np

    tts = _get_xtts_model("base")
    model = tts.synthesizer.tts_model
    log.info("XTTS TTS: lang=%s, ref=%s, speed=%.2f, text_len=%d",
             language, Path(reference_audio_path).name, speed, len(text))

    # Pre-compute speaker conditioning once for all chunks
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_audio_path]
    )
    log.info("  XTTS conditioning latents computed once")

    chunks = chunk_text(text, 220)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for i, chunk in enumerate(chunks):
        out = model.inference(
            text=chunk,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
        )
        samples = np.array(out["wav"], dtype=np.float32)
        sample_rate = 24000

        max_val = np.abs(samples).max()
        if max_val > 0:
            samples = samples * (0.95 / max_val)

        pcm = (samples * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)
        combined += AudioSegment.from_wav(wav_buf)
        if i < len(chunks) - 1:
            combined += silence
        log.info("  XTTS chunk %d/%d done", i + 1, len(chunks))
        if on_progress:
            on_progress(i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def synthesize_xtts_ro(
    text: str,
    reference_audio_path: str,
    speed: float = 1.0,
    on_progress=None,
) -> io.BytesIO:
    """Synthesize Romanian text → MP3 via XTTS-v2 Romanian fine-tune."""
    import torch
    import numpy as np

    model = _get_xtts_ro_model()
    text = _normalize_romanian(text)
    log.info("XTTS-RO TTS: ref=%s, speed=%.2f, text_len=%d",
             Path(reference_audio_path).name, speed, len(text))

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_audio_path]
    )

    chunks = chunk_text(text, 220)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for i, chunk in enumerate(chunks):
        out = model.inference(
            text=chunk,
            language="ro",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            speed=speed,
        )
        samples = np.array(out["wav"], dtype=np.float32)
        sample_rate = 24000

        max_val = np.abs(samples).max()
        if max_val > 0:
            samples = samples * (0.95 / max_val)

        pcm = (samples * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)
        combined += AudioSegment.from_wav(wav_buf)
        if i < len(chunks) - 1:
            combined += silence
        log.info("  XTTS-RO chunk %d/%d done", i + 1, len(chunks))
        if on_progress:
            on_progress(i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


# ---------------------------------------------------------------------------
# SpeechT5 — multi-speaker TTS
# ---------------------------------------------------------------------------

# 10 pre-selected speaker x-vector indices from Matthijs/cmu-arctic-xvectors
SPEECHT5_SPEAKERS = {
    "clb": {"name": "CLB (Female)", "index": 7306},
    "slt": {"name": "SLT (Female)", "index": 2961},
    "rms": {"name": "RMS (Male)", "index": 1089},
    "bdl": {"name": "BDL (Male)", "index": 4446},
    "ksp": {"name": "KSP (Male)", "index": 6529},
    "jmk": {"name": "JMK (Male)", "index": 8051},
    "awb": {"name": "AWB (Male, Scottish)", "index": 5393},
    "fem1": {"name": "Female Speaker 1", "index": 1500},
    "fem2": {"name": "Female Speaker 2", "index": 3200},
    "male1": {"name": "Male Speaker 1", "index": 6000},
}


def _get_speecht5():
    """Load SpeechT5 processor, model, vocoder, and speaker embeddings."""
    global _speecht5_cache
    if "model" not in _speecht5_cache:
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            from datasets import load_dataset
            import torch
        except ImportError:
            log.error("SpeechT5 deps not installed. Install with: uv sync --extra speecht5")
            raise

        _report_downloading("SpeechT5 (microsoft/speecht5_tts)")
        log.info("Loading SpeechT5 model + vocoder + speaker embeddings")
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        embeddings = {}
        for spk_id, info in SPEECHT5_SPEAKERS.items():
            embeddings[spk_id] = torch.tensor(ds[info["index"]]["xvector"]).unsqueeze(0)

        _speecht5_cache["processor"] = processor
        _speecht5_cache["model"] = model
        _speecht5_cache["vocoder"] = vocoder
        _speecht5_cache["embeddings"] = embeddings
    return _speecht5_cache


def synthesize_speecht5(text: str, speaker_id: str = "clb", on_progress=None) -> io.BytesIO:
    """Synthesize text → MP3 via SpeechT5 with speaker embedding."""
    import torch
    import numpy as np

    cache = _get_speecht5()
    processor = cache["processor"]
    model = cache["model"]
    vocoder = cache["vocoder"]
    embedding = cache["embeddings"].get(speaker_id, cache["embeddings"]["clb"])

    log.info("SpeechT5 TTS: speaker=%s, text_len=%d", speaker_id, len(text))

    chunks = chunk_text(text, 500)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for i, chunk in enumerate(chunks):
        inputs = processor(text=chunk, return_tensors="pt")
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], embedding, vocoder=vocoder)
        samples = speech.numpy()
        sample_rate = 16000

        pcm = (samples * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        wav_buf.seek(0)
        combined += AudioSegment.from_wav(wav_buf)
        if i < len(chunks) - 1:
            combined += silence
        log.info("  SpeechT5 chunk %d/%d done", i + 1, len(chunks))
        if on_progress:
            on_progress(i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


# ---------------------------------------------------------------------------
# HF Inference API (cloud)
# ---------------------------------------------------------------------------


def synthesize_hf_cloud(text: str, model_name: str, hf_token: str, on_progress=None) -> io.BytesIO:
    """Synthesize text → MP3 via HF Inference API (cloud, free tier)."""
    import requests as http_requests

    if not hf_token:
        raise ValueError("HF_TOKEN is required for Hugging Face cloud inference. Set it in .env.")

    api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    log.info("HF Cloud TTS: model=%s, text_len=%d", model_name, len(text))

    chunks = chunk_text(text, 500)
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for i, chunk in enumerate(chunks):
        resp = None
        for attempt in range(4):
            resp = http_requests.post(
                api_url,
                headers=headers,
                json={"inputs": chunk},
                timeout=60,
            )
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 503):
                wait = 2 ** attempt
                log.warning("HF Cloud rate limited (%d), retrying in %ds", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()

        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"HF Cloud API failed after retries: {resp.status_code if resp else 'no response'}")

        audio_buf = io.BytesIO(resp.content)
        try:
            segment = AudioSegment.from_file(audio_buf)
        except Exception:
            segment = AudioSegment.from_file(audio_buf, format="flac")
        combined += segment
        if i < len(chunks) - 1:
            combined += silence
        log.info("  HF Cloud chunk %d/%d done", i + 1, len(chunks))
        if on_progress:
            on_progress(i + 1, len(chunks))

    mp3_buf = io.BytesIO()
    combined.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------

LANGUAGE_REGISTRY = {
    "en": {
        "name": "English",
        "backends": ["piper", "kokoro", "supertonic", "huggingface", "xtts", "polly", "hf_cloud"],
        "default_backend": "kokoro",
    },
    "ro": {
        "name": "Romanian",
        "backends": ["xtts_ro", "huggingface", "hf_cloud"],
        "default_backend": "xtts_ro",
    },
}

# Reference audio directory
_REFERENCE_AUDIO_DIR = _PROJECT_ROOT / "reference_audio"
_REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)
_ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
_MAX_REFERENCE_MB = 10


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/reference_audio/<path:filename>")
def serve_reference_audio(filename):
    """Serve reference audio files for preview."""
    safe = Path(filename).name
    filepath = _REFERENCE_AUDIO_DIR / safe
    if not filepath.is_file():
        return _err("File not found", 404)
    return send_file(str(filepath))


@app.route("/")
def index():
    project_dir = Path(__file__).parent
    # Discover .onnx models in models/ subfolder
    models_dir = project_dir / "models"
    onnx_models: list[str] = []
    if models_dir.is_dir():
        for p in sorted(models_dir.glob("*.onnx")):
            try:
                _validate_piper_model_files(str(p))
                onnx_models.append(str(p))
            except ValueError as e:
                log.warning("Skipping invalid Piper model in UI list: %s", e)
    # Discover books (PDFs and EPUBs) in input/ folder
    input_dir = project_dir / "input"
    input_books = []
    if input_dir.is_dir():
        input_books = sorted(
            p.name for p in input_dir.glob("*")
            if p.suffix.lower() in {".pdf", ".epub"}
        )
    # Resolve the display name for the default Kokoro voice so the
    # template can pre-render the review card without waiting for JS.
    kokoro_voice_name = _KOKORO_VOICE_NAMES.get(CONFIG["KOKORO_VOICE"], CONFIG["KOKORO_VOICE"])
    return render_template(
        "index.html",
        config=CONFIG,
        kokoro_voice_name=kokoro_voice_name,
        onnx_models=onnx_models,
        input_books=input_books,
    )


def _err(msg: str, status: int = 400):
    """Return a JSON error so the fetch-based frontend can display it."""
    return jsonify({"error": msg}), status


def _get_polly_status(force_refresh: bool = False) -> dict:
    """Check Polly readiness with a short TTL cache to avoid noisy probes."""
    now = time.time()
    if not force_refresh and (now - _polly_status_cache["checked_at"]) < _POLLY_STATUS_TTL_SECONDS:
        return _polly_status_cache["result"]

    result = _probe_polly_status()
    _polly_status_cache["checked_at"] = now
    _polly_status_cache["result"] = result
    return result


def _probe_polly_status() -> dict:
    """Resolve AWS credentials and validate them with a lightweight AWS call."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            EndpointConnectionError,
            NoCredentialsError,
            PartialCredentialsError,
        )
    except ImportError:
        return {"ready": False, "note": "boto3 is not installed"}

    session = boto3.Session(region_name=CONFIG["AWS_REGION"])
    creds = session.get_credentials()
    if creds is None:
        return {"ready": False, "note": "AWS credentials not found for Polly"}

    frozen = creds.get_frozen_credentials()
    if not frozen.access_key or not frozen.secret_key:
        return {"ready": False, "note": "AWS credentials are incomplete for Polly"}

    try:
        sts = session.client(
            "sts",
            config=BotoConfig(connect_timeout=2, read_timeout=3, retries={"max_attempts": 0}),
        )
        sts.get_caller_identity()
        return {"ready": True, "note": "AWS credentials validated for Polly"}
    except NoCredentialsError:
        return {"ready": False, "note": "AWS credentials not found for Polly"}
    except PartialCredentialsError:
        return {"ready": False, "note": "AWS credentials are incomplete for Polly"}
    except EndpointConnectionError:
        return {"ready": False, "note": "Could not reach AWS STS to validate Polly credentials"}
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "ClientError")
        message = err.get("Message", "AWS rejected the credentials")
        return {"ready": False, "note": f"Polly auth failed: {code} - {message}"}
    except BotoCoreError as e:
        return {"ready": False, "note": f"Polly validation failed: {e}"}


@app.route("/api/backend-status")
def api_backend_status():
    """Return model readiness for each backend."""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    st_cache = Path(os.environ.get("SUPERTONIC_CACHE_DIR", Path.home() / ".cache" / "supertonic2"))
    models_dir = Path(__file__).parent / "models"

    def _has_hf_model(repo_id: str) -> bool:
        model_dir = cache / "hub" / f"models--{repo_id.replace('/', '--')}"
        if not model_dir.is_dir():
            return False
        return (
            any(model_dir.rglob("*.bin"))
            or any(model_dir.rglob("*.safetensors"))
            or any(model_dir.rglob("*.onnx"))
            or any(model_dir.rglob("*.pth"))
        )

    piper_ready = any(models_dir.glob("*.onnx")) if models_dir.is_dir() else False
    kokoro_ready = _has_hf_model("hexgrad/Kokoro-82M")
    supertonic_ready = st_cache.is_dir() and any(st_cache.rglob("*.onnx"))
    hf_ready = _has_hf_model("facebook/mms-tts-eng")
    xtts_ready = _has_hf_model("coqui/XTTS-v2")
    xtts_ro_ready = _has_hf_model("eduardem/xtts-v2-romanian")
    speecht5_ready = _has_hf_model("microsoft/speecht5_tts")
    hf_cloud_ready = bool(os.getenv("HF_TOKEN", "").strip())
    polly_status = _get_polly_status()

    return jsonify({
        "piper": {"ready": piper_ready, "note": "ONNX models in models/ folder"},
        "kokoro": {"ready": kokoro_ready, "note": "Downloads ~327MB from HuggingFace on first use"},
        "supertonic": {"ready": supertonic_ready, "note": "Downloads ~305MB ONNX model on first use"},
        "huggingface": {"ready": hf_ready, "note": "Downloads ~50MB from HuggingFace on first use"},
        "xtts": {"ready": xtts_ready, "note": "Downloads ~1.8GB XTTS-v2 model on first use"},
        "xtts_ro": {"ready": xtts_ro_ready, "note": "Downloads ~1.8GB Romanian fine-tune on first use"},
        "speecht5": {"ready": speecht5_ready, "note": "Downloads ~300MB SpeechT5 model on first use"},
        "hf_cloud": {"ready": hf_cloud_ready, "note": "Requires HF_TOKEN in .env"},
        "polly": polly_status,
    })


@app.route("/api/input-files")
def api_input_files():
    """Return list of PDFs and EPUBs in the input/ folder."""
    input_dir = Path(__file__).parent / "input"
    files = []
    if input_dir.is_dir():
        files = sorted(
            p.name for p in input_dir.glob("*")
            if p.suffix.lower() in {".pdf", ".epub"}
        )
    return jsonify({"files": files})


@app.route("/api/languages")
def api_languages():
    """Return the language registry for frontend filtering."""
    return jsonify(LANGUAGE_REGISTRY)


@app.route("/api/upload-reference", methods=["POST"])
def api_upload_reference():
    """Upload a reference audio file for voice cloning."""
    f = request.files.get("audio")
    if not f or not f.filename:
        return _err("No audio file provided.")
    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED_AUDIO_EXT:
        return _err(f"Invalid audio type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXT))}")
    # Check size
    f.seek(0, 2)
    size = f.tell()
    f.seek(0)
    if size > _MAX_REFERENCE_MB * 1024 * 1024:
        return _err(f"File too large ({size // 1024 // 1024}MB). Max: {_MAX_REFERENCE_MB}MB.")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(f.filename).name)
    dest = _REFERENCE_AUDIO_DIR / safe_name
    f.save(str(dest))
    log.info("Saved reference audio: %s (%d bytes)", safe_name, size)
    return jsonify({"saved": True, "filename": safe_name})


@app.route("/api/reference-voices")
def api_reference_voices():
    """List available reference audio files with durations."""
    voices = []
    if _REFERENCE_AUDIO_DIR.is_dir():
        for p in sorted(_REFERENCE_AUDIO_DIR.iterdir()):
            if p.suffix.lower() in _ALLOWED_AUDIO_EXT:
                try:
                    seg = AudioSegment.from_file(str(p))
                    duration = round(len(seg) / 1000, 1)
                except Exception:
                    duration = None
                voices.append({
                    "filename": p.name,
                    "duration_s": duration,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                })
    return jsonify({"voices": voices})


@app.route("/api/hf-speakers")
def api_hf_speakers():
    """Return available SpeechT5 speaker presets."""
    speakers = [{"id": k, "name": v["name"]} for k, v in SPEECHT5_SPEAKERS.items()]
    return jsonify({"speakers": speakers})


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


# Supertonic voice catalogue — static so the voices endpoint works without
# importing supertonic. Matches the voice_style_names the library exposes.
SUPERTONIC_VOICES: dict[str, list[dict]] = {
    "Female": [
        {"id": "F1", "name": "Female 1"},
        {"id": "F2", "name": "Female 2"},
        {"id": "F3", "name": "Female 3"},
        {"id": "F4", "name": "Female 4"},
        {"id": "F5", "name": "Female 5"},
    ],
    "Male": [
        {"id": "M1", "name": "Male 1"},
        {"id": "M2", "name": "Male 2"},
        {"id": "M3", "name": "Male 3"},
        {"id": "M4", "name": "Male 4"},
        {"id": "M5", "name": "Male 5"},
    ],
}


# Kokoro voice catalogue — single source of truth used by both the API
# endpoint and the index route (to pre-render the review card label).
KOKORO_VOICES: dict[str, list[dict]] = {
    "American Female": [
        {"id": "af_bella",   "name": "Bella"},
        {"id": "af_emma",    "name": "Emma"},
        {"id": "af_liam",    "name": "Liam"},
        {"id": "af_alice",   "name": "Alice"},
        {"id": "af_lily",    "name": "Lily"},
        {"id": "af_sarah",   "name": "Sarah"},
        {"id": "af_maya",    "name": "Maya"},
    ],
    "American Male": [
        {"id": "am_adam",    "name": "Adam"},
        {"id": "am_michael", "name": "Michael"},
        {"id": "am_brian",   "name": "Brian"},
        {"id": "am_jack",    "name": "Jack"},
        {"id": "am_david",   "name": "David"},
    ],
    "British Female": [
        {"id": "bf_emma",    "name": "Emma"},
        {"id": "bf_lily",    "name": "Lily"},
        {"id": "bf_alice",   "name": "Alice"},
        {"id": "bf_rose",    "name": "Rose"},
    ],
    "British Male": [
        {"id": "bm_james",   "name": "James"},
        {"id": "bm_oliver",  "name": "Oliver"},
        {"id": "bm_george",  "name": "George"},
    ],
}

# Flat id→name lookup derived from the catalogue above.
_KOKORO_VOICE_NAMES: dict[str, str] = {
    v["id"]: v["name"]
    for voices in KOKORO_VOICES.values()
    for v in voices
}


@app.route("/api/kokoro-voices")
def api_kokoro_voices():
    """Return available Kokoro voices grouped by accent."""
    return jsonify({"voices": KOKORO_VOICES})


@app.route("/api/supertonic-voices")
def api_supertonic_voices():
    """Return available Supertonic voices (static catalogue, no model load)."""
    return jsonify({"voices": SUPERTONIC_VOICES})


@app.route("/api/pdf-info", methods=["POST"])
def api_pdf_info():
    """Return page/chapter count for a PDF or EPUB.

    Accepts either an uploaded file or an `input_file` name from input/.
    For EPUB: "pages" are spine items (chapters).
    """
    input_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    if input_name:
        book_path = Path(__file__).parent / "input" / Path(input_name).name
        if not book_path.is_file():
            return _err(f"File not found in input/: {input_name}")
        file_ext = book_path.suffix.lower()
        file_stream = open(book_path, "rb")
        filename = book_path.name
    elif pdf_file:
        file_ext = Path(pdf_file.filename).suffix.lower()
        file_stream = pdf_file.stream
        filename = pdf_file.filename
    else:
        return _err("No file provided.")

    try:
        if file_ext == ".epub":
            return _api_epub_info(file_stream, filename)
        else:  # .pdf
            return _api_pdf_info_impl(file_stream, filename)
    finally:
        if input_name:
            file_stream.close()


def _detect_pdf_chapters(doc) -> list[dict]:
    """Detect chapters in a PDF using multiple strategies.

    Accepts a fitz Document object.
    Returns a list of dicts: [{"title": str, "page": int (1-based), "depth": int}, ...]
    """
    chapters: list[dict] = []

    # 1) Try PDF outline/bookmarks via fitz get_toc()
    # Returns [[level, title, page], ...] — flat list, no recursion needed
    toc = doc.get_toc(simple=True)
    if toc:
        for level, title, page in toc:
            if title and page >= 1:
                chapters.append({"title": title.strip(), "page": page, "depth": level - 1})

    # 2) Detect "Book Title  |  Section Name" page headers
    if not chapters:
        header_re = re.compile(r'^.{3,}\s*\|\s*(.+)$')
        chapter_label_re = re.compile(
            r'^(?:CHAPTER\s+\d+|PROLOGUE|EPILOGUE|BIBLIOGRAPHY'
            r'|INTRODUCTION|CONCLUSION|APPENDIX)\s*$',
            re.IGNORECASE,
        )
        prev_section = None
        for i in range(len(doc)):
            txt = (doc.load_page(i).get_text() or "")[:500].strip()
            first_line = txt.split("\n")[0].strip() if txt else ""
            m = header_re.match(first_line)
            if m:
                section = m.group(1).strip().rstrip(".")
                if section != prev_section:
                    title = section
                    lines = txt.split("\n")[1:]
                    label_idx = None
                    for j, line in enumerate(lines):
                        if chapter_label_re.match(line.strip()):
                            label_idx = j
                            break
                    if label_idx is not None:
                        label = lines[label_idx].strip()
                        label_norm = " ".join(label.split()).upper()
                        subtitle_parts = []
                        for sl in lines[label_idx + 1:]:
                            sl_s = sl.strip()
                            if not sl_s:
                                break
                            sl_norm = " ".join(sl_s.split()).upper()
                            if sl_norm.startswith(label_norm):
                                break
                            subtitle_parts.append(sl_s)
                        if subtitle_parts:
                            title = label.title() + ": " + " ".join(subtitle_parts)
                    prev_section = section
                    chapters.append({"title": title, "page": i + 1, "depth": 0})

    # 3) Parse a "TABLE OF CONTENTS" page
    if not chapters:
        toc_page_text = None
        for i in range(min(20, len(doc))):
            txt = (doc.load_page(i).get_text() or "").strip()
            if re.search(r'TABLE\s+OF\s+CONTENTS|CONTENTS', txt, re.IGNORECASE):
                toc_page_text = txt
                break
        if toc_page_text:
            toc_entry_re = re.compile(
                r'^((?:Prologue|Epilogue|Chapter\s+\d+|Part\s+[\dIVXLCivxlc]+'
                r'|Introduction|Conclusion|Appendix|Bibliography'
                r'|Acknowledgements?)[\s:—–\-].*)$',
                re.IGNORECASE | re.MULTILINE,
            )
            entries = [m.group(1).strip() for m in toc_entry_re.finditer(toc_page_text)]
            standalone_re = re.compile(
                r'^(Bibliography[\w\s]*)$', re.IGNORECASE | re.MULTILINE,
            )
            for m in standalone_re.finditer(toc_page_text):
                val = m.group(1).strip()
                if val not in entries:
                    entries.append(val)
            for entry in entries:
                key = entry.split('\n')[0].strip()[:60]
                for pi in range(len(doc)):
                    txt = (doc.load_page(pi).get_text() or "")[:500]
                    if key in txt and pi + 1 not in [c["page"] for c in chapters]:
                        chapters.append({"title": entry, "page": pi + 1, "depth": 0})
                        break

    # 4) Heuristic scan for chapter headings in page text
    if not chapters:
        chapter_re = re.compile(
            r'^(?:chapter|part|section|prologue|epilogue|introduction|conclusion)'
            r'(?:\s+[\dIVXLCivxlc]+)?',
            re.IGNORECASE,
        )
        for i in range(len(doc)):
            txt = (doc.load_page(i).get_text() or "")[:300].strip()
            first_line = txt.split("\n")[0].strip() if txt else ""
            if first_line and chapter_re.match(first_line):
                chapters.append({"title": first_line, "page": i + 1, "depth": 0})

    # 5) Last-resort heuristic: split on 3+ blank lines within full document text
    if not chapters:
        all_text = "\n".join(doc.load_page(i).get_text() or "" for i in range(len(doc)))
        sections = re.split(r'\n{3,}', all_text)
        sections = [s.strip() for s in sections if s.strip()]
        if len(sections) >= 2:
            for idx, section in enumerate(sections):
                title_line = section.split("\n")[0].strip()[:80] or f"Section {idx + 1}"
                chapters.append({"title": title_line, "page": idx + 1, "depth": 0})

    return chapters


def _remove_chapter_overlap(prev_text: str, curr_text: str, check_lines: int = 20) -> str:
    """Remove lines duplicated between the end of prev_text and start of curr_text.

    PyMuPDF page ranges can overlap at chapter boundaries. This finds the longest
    matching tail/head sequence (up to *check_lines*) and strips it from *prev_text*.
    """
    prev_lines = prev_text.splitlines()
    curr_lines = curr_text.splitlines()
    max_overlap = min(len(prev_lines), len(curr_lines), check_lines)
    for size in range(max_overlap, 0, -1):
        if prev_lines[-size:] == curr_lines[:size]:
            return "\n".join(prev_lines[:-size])
    return prev_text


def _api_pdf_info_impl(file_stream, filename: str):
    """PDF-specific page/chapter info extraction."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    total = len(doc)
    pages = []
    for i in range(total):
        page = doc.load_page(i)
        txt = (page.get_text() or "").strip()
        pages.append({
            "page": i + 1,
            "has_text": len(txt) > 20,
            "chars": len(txt),
            "is_scanned": _is_scanned_page(page),
        })

    chapters = _detect_pdf_chapters(doc)

    return jsonify({"total_pages": total, "pages": pages, "chapters": chapters})


def _api_epub_info(file_stream, filename: str):
    """EPUB-specific info extraction."""
    try:
        entries = _load_epub_entries(file_stream)
        return jsonify({
            "total_pages": len(entries),
            "pages": [{"page": e["page"], "has_text": e["has_text"], "chars": e["chars"]} for e in entries],
            "chapters": [{"title": e["title"], "page": e["page"], "depth": 0} for e in entries],
        })
    except Exception as e:
        log.exception("EPUB info extraction failed")
        return _err(f"Failed to read EPUB: {e}")


@app.route("/progress/<job_id>")
def progress(job_id):
    """Return current conversion progress for a job."""
    with _jobs_lock:
        info = _jobs.get(job_id)
    if info is None:
        return jsonify({"phase": "unknown", "current": 0, "total": 0,
                        "paused": False, "cancelled": False})
    return jsonify({
        "phase": info.get("phase", "unknown"),
        "current": info.get("current", 0),
        "total": info.get("total", 0),
        "paused": not info["pause_event"].is_set() if "pause_event" in info else False,
        "cancelled": bool(info.get("cancel_requested")),
    })


@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def job_cancel(job_id):
    """Request cancellation of a running conversion job."""
    _request_cancel(job_id)
    return jsonify({"ok": True})


@app.route("/api/job/<job_id>/pause", methods=["POST"])
def job_pause(job_id):
    """Pause a running conversion job between chapters."""
    _request_pause(job_id)
    return jsonify({"ok": True})


@app.route("/api/job/<job_id>/resume", methods=["POST"])
def job_resume(job_id):
    """Resume a paused conversion job."""
    _request_resume(job_id)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Synthesis dispatch helpers (shared by single-file & chapter-mode convert)
# ---------------------------------------------------------------------------

def _collect_synth_params(backend: str, voice: str) -> dict:
    """Read all backend-specific form parameters into a dict (call inside request context)."""
    params: dict = {"backend": backend, "voice": voice}

    def _float_or_none(key):
        v = request.form.get(key, "").strip()
        return float(v) if v else None

    if backend == "polly":
        params["voice_id"] = voice or CONFIG["POLLY_VOICE_ID"]
        params["engine"] = request.form.get("engine", CONFIG["POLLY_ENGINE"])
    elif backend == "piper":
        params["model_path"] = voice or CONFIG["PIPER_MODEL"]
        params["length_scale"] = _float_or_none("length_scale")
        params["noise_scale"] = _float_or_none("noise_scale")
        params["noise_w_scale"] = _float_or_none("noise_w_scale")
        params["sentence_silence"] = _float_or_none("sentence_silence")
    elif backend == "huggingface":
        params["model_name"] = voice or CONFIG["HF_MODEL"]
    elif backend == "kokoro":
        params["voice_id"] = voice or CONFIG["KOKORO_VOICE"]
        params["speed"] = float(request.form.get("kokoro_speed", CONFIG["KOKORO_SPEED"]))
        params["lang_code"] = request.form.get("kokoro_lang", CONFIG["KOKORO_LANG"])
    elif backend == "supertonic":
        params["voice_id"] = voice or CONFIG["SUPERTONIC_VOICE"]
        params["lang"] = request.form.get("supertonic_lang", CONFIG["SUPERTONIC_LANG"])
        params["speed"] = float(request.form.get("supertonic_speed", CONFIG["SUPERTONIC_SPEED"]))
        params["silence"] = float(request.form.get("supertonic_silence", CONFIG["SUPERTONIC_SILENCE"]))
    elif backend in ("xtts", "xtts_ro"):
        ref_audio = request.form.get("reference_audio", "").strip()
        if not ref_audio:
            raise ValueError("Voice cloning requires a reference audio file. Upload one first.")
        ref_path = _REFERENCE_AUDIO_DIR / Path(ref_audio).name
        if not ref_path.is_file():
            raise ValueError(f"Reference audio not found: {ref_audio}")
        params["ref_path"] = str(ref_path)
        params["speed"] = float(request.form.get("xtts_speed", "1.0"))
        if backend == "xtts":
            params["xtts_lang"] = request.form.get("xtts_language", "en")
    elif backend == "speecht5":
        speaker_id = request.form.get("speecht5_speaker", "clb")
        if speaker_id not in SPEECHT5_SPEAKERS:
            speaker_id = "clb"
        params["speaker_id"] = speaker_id
    elif backend == "hf_cloud":
        params["model_name"] = voice or CONFIG["HF_MODEL"]
        params["hf_token"] = os.getenv("HF_TOKEN", "")
    return params


def _do_synthesis(text: str, backend: str, params: dict, on_progress=None) -> dict:
    """Run TTS synthesis and return {mp3_buf, provider_detail, prosody_info, ...}.

    Raises ValueError for user-facing validation errors, other exceptions bubble up.
    """
    prosody_info: dict = {}
    provider_detail = "default"
    polly_chars_billed = None
    polly_cost_usd = None

    if backend == "polly":
        provider_detail = f"{params['voice_id']}_{params['engine']}"
        mp3_buf, polly_chars_billed = synthesize_polly(
            text, params["voice_id"], params["engine"], on_progress=on_progress)
        rate = _POLLY_RATES.get(params["engine"].lower(), _POLLY_RATES["neural"])
        polly_cost_usd = polly_chars_billed * rate / 1_000_000
        prosody_info = {"engine": params["engine"]}

    elif backend == "piper":
        provider_detail = Path(params["model_path"]).name
        mp3_buf = synthesize_piper(
            text, params["model_path"],
            length_scale=params["length_scale"],
            noise_scale=params["noise_scale"],
            noise_w_scale=params["noise_w_scale"],
            sentence_silence=params["sentence_silence"],
            on_progress=on_progress,
        )
        prosody_info = {
            "ls": params["length_scale"], "ns": params["noise_scale"],
            "nw": params["noise_w_scale"], "ss": params["sentence_silence"],
        }

    elif backend == "huggingface":
        provider_detail = params["model_name"]
        mp3_buf = synthesize_huggingface(text, params["model_name"], on_progress=on_progress)

    elif backend == "kokoro":
        provider_detail = params["voice_id"]
        mp3_buf = synthesize_kokoro(
            text, params["voice_id"], params["speed"], params["lang_code"],
            on_progress=on_progress)
        prosody_info = {"spd": params["speed"], "lang": params["lang_code"]}

    elif backend == "supertonic":
        provider_detail = params["voice_id"]
        mp3_buf = synthesize_supertonic(
            text, params["voice_id"], params["lang"],
            speed=params["speed"], silence_duration=params["silence"],
            on_progress=on_progress)
        prosody_info = {"spd": params["speed"], "sil": params["silence"], "lang": params["lang"]}

    elif backend == "xtts":
        provider_detail = f"xtts-v2_{params['xtts_lang']}"
        mp3_buf = synthesize_xtts(
            text, params["xtts_lang"], params["ref_path"],
            speed=params["speed"], on_progress=on_progress)
        prosody_info = {"spd": params["speed"], "lang": params["xtts_lang"]}

    elif backend == "xtts_ro":
        provider_detail = "xtts-v2-romanian"
        mp3_buf = synthesize_xtts_ro(
            text, params["ref_path"], speed=params["speed"], on_progress=on_progress)
        prosody_info = {"spd": params["speed"], "lang": "ro"}

    elif backend == "speecht5":
        provider_detail = f"speecht5_{params['speaker_id']}"
        mp3_buf = synthesize_speecht5(text, params["speaker_id"], on_progress=on_progress)

    elif backend == "hf_cloud":
        provider_detail = f"hf-cloud_{params['model_name']}"
        mp3_buf = synthesize_hf_cloud(
            text, params["model_name"], params["hf_token"], on_progress=on_progress)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    result = {
        "mp3_buf": mp3_buf,
        "provider_detail": provider_detail,
        "prosody_info": prosody_info,
    }
    if polly_chars_billed is not None:
        result["polly_chars_billed"] = polly_chars_billed
        result["polly_cost_usd"] = polly_cost_usd
    return result


def _build_pdf_chapter_texts(file_stream) -> list[tuple[str, str]]:
    """Return chapter titles and preprocessed text for a PDF."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    total_pages = len(doc)
    chapters = _detect_pdf_chapters(doc)

    if not chapters:
        raise ValueError(
            "No chapters detected in this PDF. Chapter mode requires detectable chapter boundaries."
        )

    segments: list[tuple[str, int, int]] = []
    first_ch_page = chapters[0]["page"]
    if first_ch_page > 1:
        segments.append(("Intro", 0, first_ch_page - 2))

    for idx, ch in enumerate(chapters):
        start = ch["page"] - 1
        end = chapters[idx + 1]["page"] - 2 if idx < len(chapters) - 1 else total_pages - 1
        segments.append((ch["title"], start, end))

    header_patterns = _build_pdf_header_patterns(doc)
    chapter_texts: list[tuple[str, str]] = []
    for title, start_pg, end_pg in segments:
        parts = []
        for pi in range(start_pg, end_pg + 1):
            page_text = _extract_page_text_fitz(doc.load_page(pi))
            parts.append(_strip_page_headers(page_text, header_patterns))
        chapter_texts.append((title, preprocess_text_for_speech("\n".join(parts))))

    # Remove text duplicated between adjacent chapter boundaries
    for i in range(len(chapter_texts) - 1):
        title_a, text_a = chapter_texts[i]
        _, text_b = chapter_texts[i + 1]
        chapter_texts[i] = (title_a, _remove_chapter_overlap(text_a, text_b))

    return chapter_texts


def _build_epub_chapter_texts(file_stream) -> list[tuple[str, str]]:
    """Return chapter titles and preprocessed text for an EPUB."""
    entries = _load_epub_entries(file_stream)
    chapter_texts = []
    for entry in entries:
        if not entry["text"].strip():
            continue
        chapter_texts.append((entry["title"], preprocess_text_for_speech(entry["text"])))
    return chapter_texts


def _convert_chapters(file_stream, source_filename, input_file_name,
                      backend, voice, job_id, save_to_output, t0):
    """Convert a document into one MP3 per chapter, returned as a ZIP."""
    import zipfile as zf

    file_ext = Path(source_filename).suffix.lower()
    _report_progress(job_id, 0, 1, phase="extracting")

    try:
        if file_ext == ".pdf":
            chapter_texts = _build_pdf_chapter_texts(file_stream)
        elif file_ext == ".epub":
            chapter_texts = _build_epub_chapter_texts(file_stream)
        else:
            return _err(f"Chapter mode does not support '{file_ext}' files.")
    except Exception as e:
        return _err(f"Failed to read document: {e}")
    finally:
        if input_file_name:
            file_stream.close()

    if not chapter_texts:
        return _err("No chapters detected in this document.")

    log.info("Chapter mode: %d segments detected", len(chapter_texts))
    for i, (title, text) in enumerate(chapter_texts):
        log.info("  [%d] %s (%d chars)", i, title, len(text))

    # Collect synthesis params (while still in request context)
    try:
        synth_params = _collect_synth_params(backend, voice)
    except ValueError as e:
        _clear_job(job_id)
        return _err(str(e))

    _job_context.job_id = job_id  # allow lazy loaders to report download progress

    # Synthesize each segment — write each MP3 to disk immediately so
    # completed chapters survive even if a later chapter fails or the
    # process crashes.  For save_to_output mode the files go straight
    # to the final destination; for download mode they go to a temp dir
    # that is ZIPped at the end.
    book_slug = _slug_token(Path(source_filename).stem, max_len=60)
    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    if save_to_output:
        output_dir = Path(__file__).parent / "output"
        chapter_dir = output_dir / f"{iso_ts}_{book_slug}_chapters"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        work_dir = chapter_dir
        tmp_obj = None  # nothing to clean up
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="audiobook_chapters_")
        work_dir = Path(tmp_obj.name)

    mp3_count = 0
    mp3_filenames: list[str] = []

    try:
        for seg_idx, (title, text) in enumerate(chapter_texts):
            # Check for cancellation before each chapter
            if _check_cancelled(job_id):
                log.info("Chapter conversion cancelled at segment %d", seg_idx)
                break
            # Block here if the job has been paused
            _wait_if_paused(job_id)

            _report_progress(job_id, seg_idx, len(chapter_texts),
                             phase="synthesizing")

            if not text.strip():
                log.info("  Skipping empty chapter: %s", title)
                continue

            log.info("  Synthesizing chapter %d/%d: %s (%d chars)",
                     seg_idx + 1, len(chapter_texts), title, len(text))

            try:
                result = _do_synthesis(text, backend, synth_params,
                                       on_progress=None)
            except Exception as e:
                log.exception("TTS failed for chapter: %s", title)
                _clear_job(job_id)
                return _err(f"TTS error on chapter '{title}': {e}", 500)

            # Write chapter MP3 to disk immediately
            ch_filename = f"{seg_idx:02d}_{_slug_token(title, max_len=80)}.mp3"
            buf = result["mp3_buf"]
            buf.seek(0)
            (work_dir / ch_filename).write_bytes(buf.read())
            del buf  # free memory right away
            mp3_filenames.append(ch_filename)
            mp3_count += 1
            log.info("  Wrote %s (%d/%d)", ch_filename,
                     mp3_count, len(chapter_texts))

        if not mp3_count:
            _clear_job(job_id)
            return _err("No chapters produced any audio.")

        elapsed = time.time() - t0
        log.info("Chapter mode done in %.1fs — %d MP3s", elapsed, mp3_count)

        if save_to_output:
            _clear_job(job_id)
            return jsonify({
                "saved": True,
                "filename": chapter_dir.name,
                "path": str(chapter_dir),
                "chapter_count": mp3_count,
            })

        # Download mode: ZIP from the temp files on disk
        _report_progress(job_id, len(chapter_texts), len(chapter_texts),
                         phase="encoding")
        zip_buf = io.BytesIO()
        with zf.ZipFile(zip_buf, "w", zf.ZIP_DEFLATED) as zipf:
            for fname in mp3_filenames:
                zipf.write(work_dir / fname, fname)
        zip_buf.seek(0)

        zip_filename = (f"{iso_ts}_{book_slug}_chapters_"
                        f"{_slug_token(backend)}.zip")
        _clear_job(job_id)
        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name=zip_filename,
        )
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()


@app.route("/api/voice-test", methods=["POST"])
def voice_test():
    """Synthesize a short preview for voice testing.

    Accepts form fields: backend, voice, text (max 500 chars).
    Rate-limited to one request per 3 seconds per session.
    Returns an MP3 audio response.
    """
    import time as _time

    backend = request.form.get("backend", "").strip()
    voice = request.form.get("voice", "").strip()
    text = request.form.get("text", "").strip()

    if not text:
        return _err("Please provide text to synthesize.", 400)
    if len(text) > 500:
        return _err("Preview text must be 500 characters or fewer.", 400)

    ALLOWED_BACKENDS = {
        "polly", "piper", "huggingface", "kokoro", "supertonic",
        "xtts", "xtts_ro", "speecht5", "hf_cloud",
    }
    if backend not in ALLOWED_BACKENDS:
        return _err(f"Unknown backend: '{backend}'", 400)

    # Rate limiting: max 1 request per 3 seconds (server-side per session)
    from flask import session
    now = _time.time()
    last = session.get("last_voice_test", 0)
    if now - last < 3.0:
        remaining = round(3.0 - (now - last), 1)
        return _err(f"Please wait {remaining}s before generating another preview.", 429)
    session["last_voice_test"] = now

    try:
        synth_params = _collect_synth_params(backend, voice)
    except ValueError as e:
        return _err(str(e), 400)

    try:
        result = _do_synthesis(text, backend, synth_params)
    except ValueError as e:
        return _err(str(e), 400)
    except Exception as e:
        log.exception("Voice test synthesis failed")
        return _err(f"Synthesis error: {e}", 500)

    mp3_buf = result["mp3_buf"]
    mp3_buf.seek(0)
    return send_file(mp3_buf, mimetype="audio/mpeg")


@app.route("/convert", methods=["POST"])
def convert():
    t0 = time.time()

    # --- Collect & validate form data ---
    job_id = request.form.get("job_id", "")
    save_to_output = request.form.get("save_to_output") == "1"
    input_file_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    # Determine file source: input/ folder file or uploaded file
    if input_file_name:
        book_path = Path(__file__).parent / "input" / Path(input_file_name).name
        if not book_path.is_file():
            return _err(f"File not found in input/: {input_file_name}")
        file_stream = open(book_path, "rb")
        source_filename = book_path.name
    elif pdf_file and pdf_file.filename:
        ext = Path(pdf_file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _err(f"Invalid file type '{ext}'. Only PDF and EPUB files are accepted.")
        
        # Check magic byte only for PDFs
        if ext == ".pdf":
            header = pdf_file.stream.read(4)
            pdf_file.stream.seek(0)
            if header[:4] != PDF_MAGIC:
                return _err("The uploaded file does not appear to be a valid PDF.")
        
        file_stream = pdf_file.stream
        source_filename = pdf_file.filename
    else:
        return _err("Please upload a file or select one from the input folder.")

    page_range = request.form.get("page_range", "all").strip()
    backend = request.form.get("backend", CONFIG["DEFAULT_BACKEND"])
    voice = request.form.get("voice", "").strip()

    if backend not in ("polly", "piper", "huggingface", "kokoro", "supertonic", "xtts", "xtts_ro", "hf_cloud", "speecht5"):
        return _err(f"Unknown backend: {backend}")

    log.info(
        "Request: backend=%s, pages=%s, voice=%s, file=%s, save_to_output=%s",
        backend,
        page_range,
        voice or "(default)",
        source_filename,
        save_to_output,
    )

    # --- Chapter mode: one MP3 per chapter → ZIP ---
    chapter_mode = request.form.get("chapter_mode") == "1"
    if chapter_mode:
        return _convert_chapters(
            file_stream, source_filename, input_file_name,
            backend, voice, job_id, save_to_output, t0,
        )

    # --- Extract text ---
    _report_progress(job_id, 0, 1, phase="extracting")
    try:
        file_ext = Path(source_filename).suffix.lower()
        if file_ext == ".epub":
            text, label = extract_text_from_epub(file_stream, page_range)
        else:  # .pdf
            text, label = extract_text_from_pdf(file_stream, page_range)
    except ValueError as e:
        return _err(str(e))
    finally:
        if input_file_name:
            file_stream.close()

    if not text.strip():
        return _err(
            "No extractable text found in the selected pages. "
            "The file may contain scanned images instead of text."
        )

    if len(text) > MAX_TEXT_CHARS:
        return _err(
            f"Extracted text is too long ({len(text):,} chars). "
            f"Please select fewer pages (limit: {MAX_TEXT_CHARS:,} chars)."
        )

    log.info("Extracted text: %d chars from %s", len(text), label)

    # --- Collect synthesis parameters from form (once) ---
    synth_params = _collect_synth_params(backend, voice)

    # --- Synthesize ---
    _report_progress(job_id, 0, 1, phase="synthesizing")
    _job_context.job_id = job_id  # allow lazy loaders to report download progress
    on_prog = _make_on_progress(job_id)
    try:
        result = _do_synthesis(text, backend, synth_params, on_prog)
    except JobCancelledError:
        log.info("Single-file conversion cancelled (job %s)", job_id)
        _clear_job(job_id)
        return _err("Conversion was cancelled.", 400)
    except ValueError as e:
        _clear_job(job_id)
        return _err(str(e))
    except Exception as e:
        log.exception("TTS synthesis failed")
        _clear_job(job_id)
        return _err(f"TTS error ({backend}): {e}", 500)

    mp3_buf = result["mp3_buf"]
    provider_detail = result["provider_detail"]
    prosody_info = result["prosody_info"]
    polly_chars_billed = result.get("polly_chars_billed")
    polly_cost_usd = result.get("polly_cost_usd")

    _report_progress(job_id, 1, 1, phase="encoding")
    elapsed = time.time() - t0
    log.info("Done in %.1fs — serving MP3", elapsed)

    filename = build_output_filename(source_filename, label, backend, provider_detail, prosody_info)

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
        _clear_job(job_id)
        return jsonify({"saved": True, "filename": filename, "path": str(output_path), **extra})

    _clear_job(job_id)
    return send_file(
        mp3_buf,
        mimetype="audio/mpeg",
        as_attachment=True,
        download_name=filename,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app.run(host="0.0.0.0", port=1234, debug=True)


if __name__ == "__main__":
    main()
