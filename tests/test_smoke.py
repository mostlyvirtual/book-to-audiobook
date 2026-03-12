"""Fast regression tests for book-to-audiobook."""

import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


@pytest.fixture(scope="session")
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _make_epub_bytes(chapters: list[tuple[str, str]]) -> bytes:
    """Build a minimal EPUB with one XHTML file per chapter."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )

        manifest = []
        spine = []
        for idx, (title, body) in enumerate(chapters, start=1):
            href = f"chapter{idx}.xhtml"
            item_id = f"chap{idx}"
            manifest.append(
                f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>'
            )
            spine.append(f'<itemref idref="{item_id}"/>')
            zf.writestr(
                f"OEBPS/{href}",
                f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>{title}</title></head>
  <body><h1>{title}</h1><p>{body}</p></body>
</html>""",
            )

        zf.writestr(
            "OEBPS/content.opf",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="BookId" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Sample Book</dc:title>
  </metadata>
  <manifest>{''.join(manifest)}</manifest>
  <spine>{''.join(spine)}</spine>
</package>""",
        )
    return buf.getvalue()


def _fake_mp3_result(*_args, **_kwargs):
    return {
        "mp3_buf": io.BytesIO(b"fake-mp3"),
        "provider_detail": "fake",
        "prosody_info": {},
    }


class FakeAudioSegment:
    silent_calls = []

    def __init__(self, duration=0):
        self.duration = duration

    def __iadd__(self, other):
        self.duration += getattr(other, "duration", 0)
        return self

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def silent(cls, duration=0):
        cls.silent_calls.append(duration)
        return cls(duration=duration)

    @classmethod
    def from_wav(cls, _buf):
        return cls(duration=1)

    def export(self, out_buf, format="mp3", bitrate="128k"):
        out_buf.write(b"fake-mp3-data")
        out_buf.seek(0)


class FakeSupertonic:
    sample_rate = 44100
    voice_style_names = ["F1"]

    def get_voice_style(self, style_name):
        return style_name

    def synthesize(self, *args, **kwargs):
        return [0.1, -0.1, 0.2]


class TestAPIEndpoints:
    def test_backend_status_endpoint_uses_polly_probe(self, client, monkeypatch):
        monkeypatch.setattr(
            app_module,
            "_get_polly_status",
            lambda force_refresh=False: {"ready": False, "note": "AWS credentials not found for Polly"},
        )
        resp = client.get("/api/backend-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["polly"]["ready"] is False
        assert "credentials" in data["polly"]["note"].lower()

    def test_input_files_endpoint(self, client):
        resp = client.get("/api/input-files")
        assert resp.status_code == 200
        assert isinstance(resp.get_json()["files"], list)

    def test_languages_endpoint(self, client):
        resp = client.get("/api/languages")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "en" in data and "ro" in data

    def test_index_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Split into chapter files" in resp.data


class TestTextPreprocessing:
    def test_preserves_bracketed_book_content(self):
        text = "He whispered [aside] and the crowd answered [applause]."
        cleaned = app_module.preprocess_text_for_speech(text)
        assert "[aside]" in cleaned
        assert "[applause]" in cleaned

    def test_removes_numeric_and_ref_citations(self):
        text = "Claim one[1] and claim two [Refs 2, 4] remain."
        cleaned = app_module.preprocess_text_for_speech(text)
        assert "[1]" not in cleaned
        assert "[Refs 2, 4]" not in cleaned


class TestEpubSupport:
    def test_load_epub_entries_extracts_titles(self):
        epub_bytes = _make_epub_bytes(
            [("Opening", "First body text."), ("Ending", "Second body text.")]
        )
        entries = app_module._load_epub_entries(io.BytesIO(epub_bytes))
        assert [entry["title"] for entry in entries] == ["Opening", "Ending"]
        assert all(entry["has_text"] for entry in entries)

    def test_epub_info_endpoint_lists_chapters(self, client):
        epub_bytes = _make_epub_bytes([("Opening", "Body one"), ("Ending", "Body two")])
        resp = client.post(
            "/api/pdf-info",
            data={"pdf": (io.BytesIO(epub_bytes), "book.epub")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_pages"] == 2
        assert [ch["title"] for ch in data["chapters"]] == ["Opening", "Ending"]

    def test_convert_chapters_epub_downloads_zip(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        epub_bytes = _make_epub_bytes([("Opening", "Body one"), ("Ending", "Body two")])
        resp = client.post(
            "/convert",
            data={
                "backend": "kokoro",
                "chapter_mode": "1",
                "pdf": (io.BytesIO(epub_bytes), "book.epub"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.mimetype == "application/zip"
        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            assert zf.namelist() == ["00_Opening.mp3", "01_Ending.mp3"]

    def test_convert_chapters_epub_save_to_output(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_do_synthesis", _fake_mp3_result)
        epub_bytes = _make_epub_bytes([("Opening", "Body one"), ("Ending", "Body two")])
        output_root = Path(app_module.__file__).parent / "output"
        before = {p.name for p in output_root.iterdir()}

        resp = client.post(
            "/convert",
            data={
                "backend": "kokoro",
                "chapter_mode": "1",
                "save_to_output": "1",
                "pdf": (io.BytesIO(epub_bytes), "book.epub"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["saved"] is True
        assert data["chapter_count"] == 2

        created = output_root / data["filename"]
        assert created.is_dir()
        assert sorted(p.name for p in created.iterdir()) == ["00_Opening.mp3", "01_Ending.mp3"]
        shutil.rmtree(created)
        after = {p.name for p in output_root.iterdir()}
        assert before == after


class TestPiperCliFallback:
    def test_cli_includes_noise_flags(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "--help" in cmd:
                return type("Proc", (), {"stdout": "--noise-scale --noise-w", "stderr": ""})()
            output_path = Path(cmd[cmd.index("--output_file") + 1])
            output_path.write_bytes(b"RIFFfake")
            return type("Proc", (), {"stdout": "", "stderr": ""})()

        monkeypatch.setattr(app_module, "_piper_cli_noise_flags_supported", None)
        monkeypatch.setattr("subprocess.run", fake_run)
        result = app_module._synthesize_piper_cli(
            "hello",
            "voice.onnx",
            "voice.onnx.json",
            length_scale=1.2,
            sentence_silence=0.5,
            noise_scale=0.7,
            noise_w_scale=0.9,
        )
        assert isinstance(result, io.BytesIO)
        synth_cmd = calls[-1]
        assert "--noise-scale" in synth_cmd
        assert "0.7" in synth_cmd
        assert "--noise-w" in synth_cmd
        assert "0.9" in synth_cmd

    def test_cli_raises_when_noise_flags_unsupported(self, monkeypatch):
        monkeypatch.setattr(app_module, "_piper_cli_noise_flags_supported", False)
        with pytest.raises(ValueError, match="does not support"):
            app_module._synthesize_piper_cli("hello", "voice.onnx", "voice.onnx.json")


class TestSupertonicPauseHandling:
    def test_supertonic_uses_configured_pause_duration(self, monkeypatch):
        FakeAudioSegment.silent_calls = []
        monkeypatch.setattr(app_module, "_get_supertonic_tts", lambda: FakeSupertonic())
        monkeypatch.setattr(app_module, "AudioSegment", FakeAudioSegment)

        buf = app_module.synthesize_supertonic(
            "one two three four five six seven eight nine ten " * 400,
            voice_name="F1",
            lang="en",
            silence_duration=0.75,
        )

        assert isinstance(buf, io.BytesIO)
        assert 750 in FakeAudioSegment.silent_calls


class TestPollyStatusCaching:
    def test_get_polly_status_uses_cache(self, monkeypatch):
        calls = []
        times = iter([100.0, 100.0, 120.0, 120.0])
        monkeypatch.setattr(app_module, "_polly_status_cache", {"checked_at": 0.0, "result": {}})
        monkeypatch.setattr(app_module, "_POLLY_STATUS_TTL_SECONDS", 60)
        monkeypatch.setattr(app_module.time, "time", lambda: next(times))

        def fake_probe():
            calls.append(True)
            return {"ready": True, "note": "ok"}

        monkeypatch.setattr(app_module, "_probe_polly_status", fake_probe)
        first = app_module._get_polly_status()
        second = app_module._get_polly_status()
        assert first == second
        assert len(calls) == 1


class TestUtilityFunctions:
    def test_validate_missing_model_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            app_module._validate_piper_model_files(str(tmp_path / "nonexistent.onnx"))

    def test_validate_bad_json_raises(self, tmp_path):
        model = tmp_path / "bad.onnx"
        model.write_bytes(b"\x00")
        sidecar = tmp_path / "bad.onnx.json"
        sidecar.write_text("not json at all")
        with pytest.raises(ValueError, match="invalid"):
            app_module._validate_piper_model_files(str(model))

    def test_filename_prosody_none_values_skipped(self):
        name = app_module.build_output_filename(
            "book.pdf",
            "p1",
            "piper",
            "amy",
            prosody={"ls": None, "ns": 0.5, "nw": None, "ss": None},
        )
        assert "ns0.5" in name
        assert "ls" not in name

    def test_slug_token(self):
        assert app_module._slug_token("Hello World!") == "Hello-World"
        assert app_module._slug_token("") == "na"
        assert app_module._slug_token("a" * 100, max_len=10) == "a" * 10
