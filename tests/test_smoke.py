"""Smoke tests for book-to-audiobook.

Validates that all TTS backends load, models initialise, and API
endpoints respond correctly.  These are the same checks that were
previously run ad-hoc from the terminal, now codified as a repeatable
pytest suite.

Run:  pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, CONFIG  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Flask test client (no real server needed)."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Import checks — backends can be imported without error
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_piper(self):
        import piper  # noqa: F401
        from piper import PiperVoice  # noqa: F401

    def test_import_supertonic(self):
        import supertonic  # noqa: F401
        from supertonic import TTS  # noqa: F401

    def test_import_kokoro(self):
        import kokoro  # noqa: F401
        from kokoro import KPipeline  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Model loading — backends initialise with real weights
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_piper_voice_loads(self):
        from app import _get_piper_voice
        voice = _get_piper_voice(CONFIG["PIPER_MODEL"])
        assert voice is not None
        assert hasattr(voice, "synthesize")

    def test_supertonic_tts_loads(self):
        from app import _get_supertonic_tts
        tts = _get_supertonic_tts()
        assert tts is not None
        assert tts.sample_rate > 0

    def test_kokoro_pipeline_loads(self):
        from app import _get_kokoro_pipeline
        pipe = _get_kokoro_pipeline(CONFIG["KOKORO_LANG"])
        assert pipe is not None


# ---------------------------------------------------------------------------
# 3. Piper model validation — sidecar JSON integrity
# ---------------------------------------------------------------------------

class TestPiperModelValidation:
    def test_validate_default_model(self):
        from app import _validate_piper_model_files
        config_path = _validate_piper_model_files(CONFIG["PIPER_MODEL"])
        assert config_path.endswith(".json")
        data = json.loads(Path(config_path).read_text())
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_validate_all_model_files(self):
        from app import _validate_piper_model_files
        models_dir = Path(__file__).resolve().parent.parent / "models"
        for onnx in models_dir.glob("*.onnx"):
            config = _validate_piper_model_files(str(onnx))
            assert Path(config).is_file()

    def test_validate_missing_model_raises(self, tmp_path):
        from app import _validate_piper_model_files
        with pytest.raises(ValueError, match="not found"):
            _validate_piper_model_files(str(tmp_path / "nonexistent.onnx"))

    def test_validate_bad_json_raises(self, tmp_path):
        from app import _validate_piper_model_files
        model = tmp_path / "bad.onnx"
        model.write_bytes(b"\x00")
        sidecar = tmp_path / "bad.onnx.json"
        sidecar.write_text("not json at all")
        with pytest.raises(ValueError, match="invalid"):
            _validate_piper_model_files(str(model))


# ---------------------------------------------------------------------------
# 4. API endpoints — voice listing returns expected structure
# ---------------------------------------------------------------------------

class TestAPIEndpoints:
    def test_kokoro_voices_endpoint(self, client):
        resp = client.get("/api/kokoro-voices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "voices" in data
        voices = data["voices"]
        assert len(voices) >= 1, "Expected at least 1 voice group"
        # Check structure
        for group_name, entries in voices.items():
            assert isinstance(entries, list)
            for entry in entries:
                assert "id" in entry
                assert "name" in entry

    def test_supertonic_voices_endpoint(self, client):
        resp = client.get("/api/supertonic-voices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "voices" in data
        voices = data["voices"]
        assert len(voices) >= 1, "Expected at least 1 language group"

    def test_backend_status_endpoint(self, client):
        resp = client.get("/api/backend-status")
        assert resp.status_code == 200
        data = resp.get_json()
        for backend in ("piper", "kokoro", "supertonic", "huggingface", "polly", "xtts", "xtts_ro", "hf_cloud", "speecht5"):
            assert backend in data, f"Missing backend: {backend}"
        assert data["polly"]["ready"] is True

    def test_input_files_endpoint(self, client):
        resp = client.get("/api/input-files")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "files" in data
        assert isinstance(data["files"], list)

    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data

    def test_languages_endpoint(self, client):
        resp = client.get("/api/languages")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "en" in data
        assert "ro" in data
        assert "English" in data["en"]["name"]
        assert "backends" in data["en"]

    def test_reference_voices_endpoint(self, client):
        resp = client.get("/api/reference-voices")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "voices" in data
        assert isinstance(data["voices"], list)

    def test_hf_speakers_endpoint(self, client):
        resp = client.get("/api/hf-speakers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "speakers" in data
        assert len(data["speakers"]) >= 1
        for s in data["speakers"]:
            assert "id" in s
            assert "name" in s

    def test_upload_reference_rejects_no_file(self, client):
        resp = client.post("/api/upload-reference")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 5. Output filename — prosody tags appended correctly
# ---------------------------------------------------------------------------

class TestOutputFilename:
    def test_filename_without_prosody(self):
        from app import build_output_filename
        name = build_output_filename("book.pdf", "p1-5", "piper", "amy")
        assert name.endswith(".mp3")
        assert "piper" in name
        assert "amy" in name

    def test_filename_with_piper_prosody(self):
        from app import build_output_filename
        name = build_output_filename(
            "book.pdf", "p1-5", "piper", "amy",
            prosody={"ls": 1.2, "ns": 0.667, "nw": 0.8, "ss": 0.4},
        )
        assert "ls1.2" in name
        assert "ns0.667" in name
        assert "nw0.8" in name
        assert "ss0.4" in name
        assert name.endswith(".mp3")

    def test_filename_with_kokoro_prosody(self):
        from app import build_output_filename
        name = build_output_filename(
            "book.pdf", "all", "kokoro", "af_bella",
            prosody={"spd": 1.5, "lang": "a"},
        )
        assert "spd1.5" in name
        assert "langa" in name

    def test_filename_prosody_none_values_skipped(self):
        from app import build_output_filename
        name = build_output_filename(
            "book.pdf", "p1", "piper", "amy",
            prosody={"ls": None, "ns": 0.5, "nw": None, "ss": None},
        )
        assert "ns0.5" in name
        assert "ls" not in name

    def test_filename_empty_prosody(self):
        from app import build_output_filename
        name = build_output_filename("book.pdf", "p1", "piper", "amy", prosody={})
        # No prosody suffix when dict is empty
        assert name.count("_amy") == 1  # only the provider detail, no extra


# ---------------------------------------------------------------------------
# 6. Text preprocessing — unit tests for pipeline helpers
# ---------------------------------------------------------------------------

class TestTextPreprocessing:
    def test_chunk_text_short(self):
        from app import chunk_text
        chunks = chunk_text("Hello world", max_chars=100)
        assert chunks == ["Hello world"]

    def test_chunk_text_splits(self):
        from app import chunk_text
        text = "word " * 100  # 500 chars
        chunks = chunk_text(text.strip(), max_chars=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_slug_token(self):
        from app import _slug_token
        assert _slug_token("Hello World!") == "Hello-World"
        assert _slug_token("") == "na"
        assert _slug_token("a" * 100, max_len=10) == "a" * 10


# ---------------------------------------------------------------------------
# 7. Romanian diacritic normalization
# ---------------------------------------------------------------------------

class TestRomanianNormalization:
    def test_cedilla_to_comma(self):
        from app import _normalize_romanian
        assert _normalize_romanian("ş") == "ș"
        assert _normalize_romanian("ţ") == "ț"
        assert _normalize_romanian("Ş") == "Ș"
        assert _normalize_romanian("Ţ") == "Ț"

    def test_already_correct(self):
        from app import _normalize_romanian
        text = "București, județul Iași"
        assert _normalize_romanian(text) == text

    def test_mixed_diacritics(self):
        from app import _normalize_romanian
        assert _normalize_romanian("şcoală şi ţară") == "școală și țară"


# ---------------------------------------------------------------------------
# 8. HF pipeline cache — dict-keyed
# ---------------------------------------------------------------------------

class TestHFPipelineCache:
    def test_hf_pipelines_is_dict(self):
        from app import _hf_pipelines
        assert isinstance(_hf_pipelines, dict)
