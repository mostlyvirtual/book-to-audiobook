"""Tests for cache-aware model loading and download reporting."""
import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module


class _FakeAudioSegment:
    def __iadd__(self, other):
        return self

    def export(self, buf, format="mp3", bitrate="128k"):
        buf.write(b"ID3" + b"\x00" * 32)
        return buf

    @staticmethod
    def empty():
        return _FakeAudioSegment()

    @staticmethod
    def silent(duration=0):
        return _FakeAudioSegment()

    @staticmethod
    def from_wav(_buf):
        return _FakeAudioSegment()


def test_get_kokoro_pipeline_uses_cache_without_download(monkeypatch, caplog):
    calls = []
    download_calls = []

    class FakePipeline:
        def __init__(self, lang_code, repo_id=None, **kwargs):
            calls.append({"lang_code": lang_code, "repo_id": repo_id, **kwargs})

    monkeypatch.setattr(app_module, "_auto_install_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "_has_hf_model", lambda repo_id: True)
    monkeypatch.setattr(app_module, "_kokoro_pipelines", {})
    monkeypatch.setattr(app_module, "_report_downloading", lambda label: download_calls.append(label))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))

    caplog.set_level(logging.INFO)
    app_module._get_kokoro_pipeline("a")

    assert download_calls == []
    assert calls == [{"lang_code": "a", "repo_id": app_module._HF_KOKORO_REPO}]
    assert "Loading Kokoro TTS pipeline from cache for lang=a" in caplog.text


def test_get_kokoro_pipeline_reports_download_when_cache_missing(monkeypatch):
    download_calls = []

    class FakePipeline:
        def __init__(self, lang_code, repo_id=None, **kwargs):
            self.lang_code = lang_code
            self.repo_id = repo_id

    monkeypatch.setattr(app_module, "_auto_install_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module, "_has_hf_model", lambda repo_id: False)
    monkeypatch.setattr(app_module, "_kokoro_pipelines", {})
    monkeypatch.setattr(app_module, "_report_downloading", lambda label: download_calls.append(label))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))

    app_module._get_kokoro_pipeline("a")

    assert download_calls == ["Kokoro (lang=a)"]


def test_synthesize_kokoro_skips_voice_download_when_cached(monkeypatch):
    class DummyAudio:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([0.1, -0.1], dtype=np.float32)

    class DummyResult:
        def __init__(self):
            self.audio = DummyAudio()

    download_calls = []

    def fake_pipeline(_chunk, voice, speed):
        yield DummyResult()

    monkeypatch.setattr(app_module, "_get_kokoro_pipeline", lambda lang: fake_pipeline)
    monkeypatch.setattr(app_module, "_has_kokoro_voice", lambda voice: True)
    monkeypatch.setattr(app_module, "_report_downloading", lambda label: download_calls.append(label))
    monkeypatch.setattr(app_module, "chunk_text", lambda text, size: ["chunk-1"])
    monkeypatch.setattr(app_module, "AudioSegment", _FakeAudioSegment)

    mp3_buf = app_module.synthesize_kokoro("hello world", "am_michael", 1.0, "a")

    assert download_calls == []
    assert isinstance(mp3_buf, io.BytesIO)
