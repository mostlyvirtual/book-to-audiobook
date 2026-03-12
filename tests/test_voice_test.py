"""Tests for Phase 4: /api/voice-test endpoint."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module


@pytest.fixture(scope="session")
def client():
    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "test-secret"
    with app_module.app.test_client() as c:
        yield c


def _fake_mp3_result(text, backend, params, on_progress=None):
    buf = io.BytesIO(b"ID3" + b"\x00" * 128)
    buf.seek(0)
    return {"mp3_buf": buf, "provider_detail": "test", "prosody_info": {}}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestVoiceTestEndpoint:
    def test_returns_mp3_for_valid_request(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        monkeypatch.setattr(app_module, "_collect_synth_params",
                            lambda backend, voice: {"backend": backend, "voice": voice})
        resp = client.post(
            "/api/voice-test",
            data={"backend": "kokoro", "voice": "am_michael", "text": "Hello world."},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.mimetype == "audio/mpeg"

    def test_response_contains_audio_data(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        monkeypatch.setattr(app_module, "_collect_synth_params",
                            lambda b, v: {"backend": b, "voice": v})
        resp = client.post(
            "/api/voice-test",
            data={"backend": "kokoro", "voice": "am_michael", "text": "Test audio."},
            content_type="multipart/form-data",
        )
        assert len(resp.data) > 0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestVoiceTestRateLimit:
    def test_second_rapid_request_returns_429(self, monkeypatch):
        """Two requests within 3 seconds from the same session should be rate-limited."""
        import time

        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        monkeypatch.setattr(app_module, "_collect_synth_params",
                            lambda b, v: {"backend": b, "voice": v})

        app_module.app.config["TESTING"] = True
        app_module.app.config["SECRET_KEY"] = "rate-limit-test-secret"

        times = iter([1000.0, 1000.0, 1001.0, 1001.0])  # both within 3s window
        monkeypatch.setattr("time.time", lambda: next(times))

        with app_module.app.test_client() as c:
            with c.session_transaction() as sess:
                sess["last_voice_test"] = 999.0  # set timestamp 1s ago

            resp = c.post(
                "/api/voice-test",
                data={"backend": "kokoro", "voice": "am_michael", "text": "Second request."},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 429
        data = resp.get_json()
        assert "wait" in data["error"].lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestVoiceTestValidation:
    def test_empty_text_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        resp = client.post(
            "/api/voice-test",
            data={"backend": "kokoro", "voice": "am_michael", "text": ""},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "text" in resp.get_json()["error"].lower()

    def test_text_too_long_returns_400(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        resp = client.post(
            "/api/voice-test",
            data={"backend": "kokoro", "voice": "am_michael", "text": "x" * 501},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "500" in data["error"]

    def test_unknown_backend_returns_400(self, client):
        resp = client.post(
            "/api/voice-test",
            data={"backend": "nonexistent_tts", "voice": "", "text": "Hello."},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "nonexistent_tts" in resp.get_json()["error"]

    def test_missing_text_field_returns_400(self, client):
        resp = client.post(
            "/api/voice-test",
            data={"backend": "kokoro", "voice": "am_michael"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_all_known_backends_pass_validation(self, client, monkeypatch):
        """None of the known backends should be rejected by the allowlist check."""
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        monkeypatch.setattr(app_module, "_collect_synth_params",
                            lambda b, v: {"backend": b, "voice": v})
        known = ["polly", "piper", "huggingface", "kokoro", "supertonic",
                 "xtts", "xtts_ro", "speecht5", "hf_cloud"]
        for backend in known:
            resp = client.post(
                "/api/voice-test",
                data={"backend": backend, "voice": "", "text": "Hello."},
                content_type="multipart/form-data",
            )
            # Should not get a 400 from the allowlist check (may fail internally, that's OK)
            assert resp.status_code != 400 or "unknown backend" not in resp.get_json().get("error", "").lower(), \
                f"Backend '{backend}' was incorrectly rejected by allowlist"
