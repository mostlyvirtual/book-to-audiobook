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
import json
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
    # Kokoro
    "KOKORO_VOICE": os.getenv("KOKORO_VOICE", "af_bella"),
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
# Lazy-loaded singletons (heavy models loaded once)
# ---------------------------------------------------------------------------

_piper_voice = None
_hf_pipeline = None
_kokoro_pipelines: dict[str, object] = {}
_supertonic_tts = None


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
            log.info("Loading Kokoro TTS pipeline for lang=%s", lang_code)
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
        except ImportError:
            log.error("Kokoro not installed. Install with: pip install kokoro")
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
            log.info("Loading Supertonic TTS (may download ~305MB model on first run)")
            _supertonic_tts = TTS(auto_download=True)
        except ImportError:
            log.error("Supertonic not installed. Install with: pip install supertonic")
            raise
    return _supertonic_tts


# ---------------------------------------------------------------------------
# Text preprocessing — makes TTS output sound like an audiobook
# ---------------------------------------------------------------------------


def _join_wrapped_lines(text: str) -> str:
    """Join lines where current ends without punctuation and next starts lowercase.
    
    Fixes PDF hard-wraps like:
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
    """Remove PDF extraction artifacts: citations, page numbers, URLs.
    
    Strips: [1], [Ref], page numbers, URLs, and common metadata junk.
    """
    # Remove citation brackets: [1], [2], [Ref 45], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[Rr]ef\s*\d+\]', '', text)
    text = re.sub(r'\[[\w\s,]+\]', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove footer/header page numbers (lines that are just digits)
    text = re.sub(r'^[\s\-\d]+$', '', text, flags=re.MULTILINE)
    
    return text


def preprocess_text_for_speech(text: str) -> str:
    """Clean up PDF-extracted text for more natural TTS output.

    Pipeline:
    1. Join hard-wrapped lines (PDF line breaks)
    2. Expand abbreviations (Mr. → Mister)
    3. Convert numbers (45% → forty-five percent)
    4. Normalize ligatures, whitespace, punctuation
    5. Remove artifacts (citations, URLs, page numbers)
    6. Add sentence-ending punctuation & pause markers
    """
    # Step 1: Join wrapped lines
    text = _join_wrapped_lines(text)
    
    # Step 2: Expand abbreviations
    text = _expand_abbreviations(text)
    
    # Step 3: Convert numbers
    text = _convert_numbers(text)
    
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
    
    # Step 4: Remove artifacts (citations, URLs, page numbers)
    text = _remove_artifacts(text)

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


def extract_text_from_epub(file_stream, page_range_str: str) -> tuple[str, str]:
    """Extract and preprocess text from selected EPUB chapters.
    
    For EPUB, "pages" are actually spine items (chapters/sections).
    Supports page_range like "all", "1-5", "3".
    """
    import zipfile
    from xml.etree import ElementTree as ET
    from bs4 import BeautifulSoup
    
    try:
        # Parse EPUB as ZIP
        with zipfile.ZipFile(file_stream) as zf:
            # Step 1: Find OPF file path from container.xml
            try:
                container_xml = zf.read('META-INF/container.xml')
                root = ET.fromstring(container_xml)
                # Find the rootfile element in the OPF namespace
                rootfile = root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                if rootfile is None:
                    raise ValueError("No rootfile found in container.xml")
                opf_path = rootfile.get('full-path')
                if not opf_path:
                    raise ValueError("rootfile has no full-path attribute")
            except (KeyError, ET.ParseError) as e:
                raise ValueError(f"Invalid EPUB: could not find OPF file: {e}")
            
            # Step 2: Parse OPF to get spine order
            try:
                opf_xml = zf.read(opf_path)
                opf_root = ET.fromstring(opf_xml)
                # Define namespace
                ns = {'opf': 'http://www.idpf.org/2007/opf'}
                
                # Get manifest items
                manifest = {}
                for item in opf_root.findall('.//opf:item', ns):
                    item_id = item.get('id')
                    href = item.get('href')
                    manifest[item_id] = href
                
                # Get spine order
                spine_items = []
                for itemref in opf_root.findall('.//opf:itemref', ns):
                    item_id = itemref.get('idref')
                    if item_id in manifest:
                        spine_items.append(manifest[item_id])
            except (KeyError, ET.ParseError) as e:
                raise ValueError(f"Invalid EPUB: could not parse OPF: {e}")
            
            if not spine_items:
                raise ValueError("EPUB has no content spine")
            
            # Step 3: Parse page range
            total_items = len(spine_items)
            indices = parse_page_range(page_range_str, total_items)
            
            # Step 4: Extract text from selected spine items
            parts = []
            opf_dir = str(Path(opf_path).parent)
            
            for idx in indices:
                item_path = spine_items[idx]
                # Resolve relative path
                full_path = (Path(opf_dir) / item_path).as_posix()
                
                try:
                    content = zf.read(full_path)
                    # Parse HTML/XHTML
                    soup = BeautifulSoup(content, 'html.parser')
                    # Remove script and style elements
                    for el in soup(['script', 'style']):
                        el.decompose()
                    # Extract text
                    text = soup.get_text(separator='\n')
                    parts.append(text)
                except KeyError:
                    log.warning(f"EPUB: item not found at {full_path}")
                    continue
            
            if not parts:
                raise ValueError("No readable content found in EPUB")
            
            raw = "\n".join(parts)
            cleaned = preprocess_text_for_speech(raw)
            
            # Create label
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
    pdf_filename: str,
    page_label: str,
    backend: str,
    provider_detail: str,
    prosody: dict | None = None,
) -> str:
    """Create stable output names:

    <ISO-UTC>_<book>_<pages>_<backend>_<voice-or-model>[_prosody].mp3
    """
    iso_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    book = _slug_token(Path(pdf_filename).stem, max_len=60)
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

    except Exception as e:
        log.warning("Piper Python API failed (%s), falling back to CLI", e)
        wav_buf = _synthesize_piper_cli(text, model_path, config_path, ls, ss)

    wav_buf.seek(0)
    segment = AudioSegment.from_wav(wav_buf)
    mp3_buf = io.BytesIO()
    segment.export(mp3_buf, format="mp3", bitrate="128k")
    mp3_buf.seek(0)
    return mp3_buf


def _synthesize_piper_cli(
    text: str, model_path: str, config_path: str,
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
            "--config", config_path,
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


def synthesize_kokoro(text: str, voice: str, speed: float, lang_code: str) -> io.BytesIO:
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
            # KPipeline.__call__ is a generator yielding Result objects
            chunk_samples = []
            for result in pipe(chunk, voice=voice, speed=speed):
                if result.audio is not None:
                    chunk_samples.append(result.audio.cpu().numpy())

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
    silence = AudioSegment.silent(duration=400)
    
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
        except Exception as e:
            log.exception("Supertonic chunk synthesis failed")
            raise
    
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
    onnx_models: list[str] = []
    if models_dir.is_dir():
        for p in sorted(models_dir.glob("*.onnx")):
            try:
                _validate_piper_model_files(str(p))
                onnx_models.append(str(p))
            except ValueError as e:
                log.warning("Skipping invalid Piper model in UI list: %s", e)
    # Discover PDFs and EPUBs in input/ folder
    input_dir = project_dir / "input"
    input_files = []
    if input_dir.is_dir():
        input_files = sorted(
            p.name for p in input_dir.glob("*")
            if p.suffix.lower() in {".pdf", ".epub"}
        )
    return render_template(
        "index.html",
        config=CONFIG,
        onnx_models=onnx_models,
        input_pdfs=input_files,
    )


def _err(msg: str, status: int = 400):
    """Return a JSON error so the fetch-based frontend can display it."""
    return jsonify({"error": msg}), status


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


@app.route("/api/kokoro-voices")
def api_kokoro_voices():
    """Return available Kokoro voices grouped by accent.
    
    Kokoro has ~25 voices across American, British, and other accents.
    Format: {accent}_{name} or {accent}_{gender}_{quality}
    """
    try:
        # Hardcoded Kokoro voices (language-agnostic IDs)
        voices = {
            "American Female": [
                {"id": "af_bella", "name": "Bella"},
                {"id": "af_emma", "name": "Emma"},
                {"id": "af_liam", "name": "Liam"},
                {"id": "af_alice", "name": "Alice"},
                {"id": "af_lily", "name": "Lily"},
                {"id": "af_sarah", "name": "Sarah"},
                {"id": "af_maya", "name": "Maya"},
            ],
            "American Male": [
                {"id": "am_adam", "name": "Adam"},
                {"id": "am_michael", "name": "Michael"},
                {"id": "am_brian", "name": "Brian"},
                {"id": "am_jack", "name": "Jack"},
                {"id": "am_david", "name": "David"},
            ],
            "British Female": [
                {"id": "bf_emma", "name": "Emma"},
                {"id": "bf_lily", "name": "Lily"},
                {"id": "bf_alice", "name": "Alice"},
                {"id": "bf_rose", "name": "Rose"},
            ],
            "British Male": [
                {"id": "bm_james", "name": "James"},
                {"id": "bm_oliver", "name": "Oliver"},
                {"id": "bm_george", "name": "George"},
            ],
        }
        return jsonify({"voices": voices})
    except Exception as e:
        log.warning("Failed to fetch Kokoro voices: %s", e)
        return jsonify({"error": str(e), "voices": {}})


@app.route("/api/supertonic-voices")
def api_supertonic_voices():
    """Return available Supertonic voices (lazy-loads model on first call).
    
    Supertonic includes voices in multiple languages and styles.
    """
    try:
        tts = _get_supertonic_tts()
        # Supertonic exposes named voice styles (e.g., F1..F5, M1..M5).
        styles = list(getattr(tts, "voice_style_names", []) or [])
        by_lang = {"en": [{"id": s, "name": s} for s in styles]}
        return jsonify({"voices": by_lang})
    except Exception as e:
        log.warning("Failed to fetch Supertonic voices: %s", e)
        return jsonify({"error": str(e), "voices": {}})


@app.route("/api/pdf-info", methods=["POST"])
def api_pdf_info():
    """Return page/chapter count for a PDF or EPUB.

    Accepts either an uploaded file or an `input_file` name from input/.
    For EPUB: "pages" are spine items (chapters).
    """
    input_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    if input_name:
        pdf_path = Path(__file__).parent / "input" / Path(input_name).name
        if not pdf_path.is_file():
            return _err(f"File not found in input/: {input_name}")
        file_ext = pdf_path.suffix.lower()
        file_stream = open(pdf_path, "rb")
        filename = pdf_path.name
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


def _api_pdf_info_impl(file_stream, filename: str):
    """PDF-specific info extraction."""
    reader = PdfReader(file_stream)
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


def _api_epub_info(file_stream, filename: str):
    """EPUB-specific info extraction."""
    import zipfile
    from xml.etree import ElementTree as ET
    from bs4 import BeautifulSoup
    
    try:
        with zipfile.ZipFile(file_stream) as zf:
            # Find OPF
            try:
                container_xml = zf.read('META-INF/container.xml')
                root = ET.fromstring(container_xml)
                rootfile = root.find('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile')
                if rootfile is None:
                    return _err("Invalid EPUB: no rootfile found")
                opf_path = rootfile.get('full-path')
                if not opf_path:
                    return _err("Invalid EPUB: rootfile has no full-path")
            except (KeyError, ET.ParseError) as e:
                return _err(f"Invalid EPUB: could not parse container: {e}")
            
            # Parse OPF
            try:
                opf_xml = zf.read(opf_path)
                opf_root = ET.fromstring(opf_xml)
                ns = {'opf': 'http://www.idpf.org/2007/opf'}
                
                manifest = {}
                for item in opf_root.findall('.//opf:item', ns):
                    item_id = item.get('id')
                    href = item.get('href')
                    manifest[item_id] = href
                
                spine_items = []
                for itemref in opf_root.findall('.//opf:itemref', ns):
                    item_id = itemref.get('idref')
                    if item_id in manifest:
                        spine_items.append(manifest[item_id])
            except (KeyError, ET.ParseError):
                return _err("Invalid EPUB: could not parse OPF")
            
            if not spine_items:
                return _err("EPUB has no content")
            
            # Build "pages" (spine items)
            opf_dir = str(Path(opf_path).parent)
            pages = []
            chapters = []
            
            for i, item_path in enumerate(spine_items):
                full_path = (Path(opf_dir) / item_path).as_posix()
                has_text = False
                char_count = 0
                title = f"Chapter {i + 1}"
                
                try:
                    content = zf.read(full_path)
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Try to extract title
                    for tag in soup(['h1', 'h2', 'h3']):
                        if tag.get_text().strip():
                            title = tag.get_text().strip()
                            break
                    
                    # Count text
                    text = soup.get_text().strip()
                    has_text = len(text) > 20
                    char_count = len(text)
                except Exception:
                    pass
                
                pages.append({"page": i + 1, "has_text": has_text, "chars": char_count})
                chapters.append({"title": title, "page": i + 1, "depth": 0})
            
            return jsonify({
                "total_pages": len(spine_items),
                "pages": pages,
                "chapters": chapters
            })
    
    except Exception as e:
        log.exception("EPUB info extraction failed")
        return _err(f"Failed to read EPUB: {e}")


@app.route("/convert", methods=["POST"])
def convert():
    t0 = time.time()

    # --- Collect & validate form data ---
    save_to_output = request.form.get("save_to_output") == "1"
    input_file_name = request.form.get("input_file", "").strip()
    pdf_file = request.files.get("pdf")

    # Determine file source: input/ folder file or uploaded file
    if input_file_name:
        pdf_path = Path(__file__).parent / "input" / Path(input_file_name).name
        if not pdf_path.is_file():
            return _err(f"File not found in input/: {input_file_name}")
        file_stream = open(pdf_path, "rb")
        pdf_filename = pdf_path.name
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
        pdf_filename = pdf_file.filename
    else:
        return _err("Please upload a file or select one from the input folder.")

    page_range = request.form.get("page_range", "all").strip()
    backend = request.form.get("backend", CONFIG["DEFAULT_BACKEND"])
    voice = request.form.get("voice", "").strip()

    if backend not in ("polly", "piper", "huggingface", "kokoro", "supertonic"):
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
        file_ext = Path(pdf_filename).suffix.lower()
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

    # --- Synthesize ---
    provider_detail = "default"
    polly_chars_billed: int | None = None
    polly_cost_usd: float | None = None
    try:
        prosody_info: dict = {}

        if backend == "polly":
            voice_id = voice or CONFIG["POLLY_VOICE_ID"]
            engine = request.form.get("engine", CONFIG["POLLY_ENGINE"])
            provider_detail = f"{voice_id}_{engine}"
            mp3_buf, polly_chars_billed = synthesize_polly(text, voice_id, engine)
            rate = _POLLY_RATES.get(engine.lower(), _POLLY_RATES["neural"])
            polly_cost_usd = polly_chars_billed * rate / 1_000_000
            prosody_info = {"engine": engine}

        elif backend == "piper":
            model_path = voice or CONFIG["PIPER_MODEL"]
            provider_detail = Path(model_path).name
            # Read prosody overrides from the form (fall back to .env defaults)
            def _float_or_none(key):
                v = request.form.get(key, "").strip()
                return float(v) if v else None
            ls = _float_or_none("length_scale")
            ns = _float_or_none("noise_scale")
            nw = _float_or_none("noise_w_scale")
            ss = _float_or_none("sentence_silence")
            mp3_buf = synthesize_piper(
                text,
                model_path,
                length_scale=ls,
                noise_scale=ns,
                noise_w_scale=nw,
                sentence_silence=ss,
            )
            prosody_info = {"ls": ls, "ns": ns, "nw": nw, "ss": ss}

        elif backend == "huggingface":
            model_name = voice or CONFIG["HF_MODEL"]
            provider_detail = model_name
            mp3_buf = synthesize_huggingface(text, model_name)

        elif backend == "kokoro":
            voice_id = voice or CONFIG["KOKORO_VOICE"]
            speed = float(request.form.get("kokoro_speed", CONFIG["KOKORO_SPEED"]))
            lang_code = request.form.get("kokoro_lang", CONFIG["KOKORO_LANG"])
            provider_detail = f"{voice_id}"
            mp3_buf = synthesize_kokoro(text, voice_id, speed, lang_code)
            prosody_info = {"spd": speed, "lang": lang_code}

        elif backend == "supertonic":
            voice_id = voice or CONFIG["SUPERTONIC_VOICE"]
            lang = request.form.get("supertonic_lang", CONFIG["SUPERTONIC_LANG"])
            st_speed = float(request.form.get("supertonic_speed", CONFIG["SUPERTONIC_SPEED"]))
            st_silence = float(request.form.get("supertonic_silence", CONFIG["SUPERTONIC_SILENCE"]))
            provider_detail = f"{voice_id}"
            mp3_buf = synthesize_supertonic(text, voice_id, lang, speed=st_speed, silence_duration=st_silence)
            prosody_info = {"spd": st_speed, "sil": st_silence, "lang": lang}

    except Exception as e:
        log.exception("TTS synthesis failed")
        return _err(f"TTS error ({backend}): {e}", 500)

    elapsed = time.time() - t0
    log.info("Done in %.1fs — serving MP3", elapsed)

    filename = build_output_filename(pdf_filename, label, backend, provider_detail, prosody_info)

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
