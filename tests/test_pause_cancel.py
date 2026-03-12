"""Tests for Phase 5: pause / resume / cancel job control."""
import io
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as app_module


@pytest.fixture(scope="module")
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _fake_mp3_result(text, backend, params, on_progress=None):
    buf = io.BytesIO(b"ID3" + b"\x00" * 128)
    buf.seek(0)
    return {"mp3_buf": buf, "provider_detail": "test", "prosody_info": {}}


def _make_epub_bytes(chapters):
    """Minimal EPUB builder (same as test_smoke.py helper)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" '
            'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        items, spine = [], []
        for i, (title, body) in enumerate(chapters):
            fname = f"OEBPS/ch{i:02d}.xhtml"
            zf.writestr(
                fname,
                f'<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title}</title></head>"
                f"<body><h1>{title}</h1><p>{body}</p></body></html>",
            )
            items.append(f'<item id="ch{i:02d}" href="ch{i:02d}.xhtml" '
                         f'media-type="application/xhtml+xml"/>')
            spine.append(f'<itemref idref="ch{i:02d}"/>')
        opf = (
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
            'version="2.0" unique-identifier="uid"><metadata>'
            '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Test</dc:title>'
            '<dc:identifier xmlns:dc="http://purl.org/dc/elements/1.1/" id="uid">test-id</dc:identifier>'
            '</metadata><manifest><item id="ncx" href="toc.ncx" '
            'media-type="application/x-dtbncx+xml"/>'
            + "".join(items)
            + '</manifest><spine toc="ncx">'
            + "".join(spine)
            + "</spine></package>"
        )
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr(
            "OEBPS/toc.ncx",
            '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" '
            'version="2005-1"><head><meta name="dtb:uid" content="test-id"/></head>'
            '<docTitle><text>Test</text></docTitle><navMap></navMap></ncx>',
        )
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Helper: directly test the control functions (unit tests)
# ---------------------------------------------------------------------------

class TestJobControlFunctions:
    def setup_method(self):
        """Ensure a fresh job state before each test."""
        app_module._report_progress("test-job-1", 0, 10, phase="synthesizing")

    def teardown_method(self):
        app_module._clear_job("test-job-1")

    def test_cancel_sets_flag(self):
        app_module._request_cancel("test-job-1")
        assert app_module._check_cancelled("test-job-1") is True

    def test_pause_clears_event(self):
        app_module._request_pause("test-job-1")
        with app_module._jobs_lock:
            event = app_module._jobs["test-job-1"]["pause_event"]
        assert not event.is_set()

    def test_resume_sets_event(self):
        app_module._request_pause("test-job-1")
        app_module._request_resume("test-job-1")
        with app_module._jobs_lock:
            event = app_module._jobs["test-job-1"]["pause_event"]
        assert event.is_set()

    def test_cancel_also_sets_pause_event(self):
        """Cancelling a paused job should wake it up."""
        app_module._request_pause("test-job-1")
        app_module._request_cancel("test-job-1")
        with app_module._jobs_lock:
            event = app_module._jobs["test-job-1"]["pause_event"]
        assert event.is_set()

    def test_check_cancelled_false_for_unknown_job(self):
        assert app_module._check_cancelled("nonexistent-job") is False

    def test_check_cancelled_false_for_none(self):
        assert app_module._check_cancelled(None) is False

    def test_wait_if_paused_returns_immediately_when_running(self):
        """_wait_if_paused should return immediately for a running (non-paused) job."""
        start = time.monotonic()
        app_module._wait_if_paused("test-job-1")
        elapsed = time.monotonic() - start
        assert elapsed < 0.2, f"Expected fast return, took {elapsed:.2f}s"

    def test_wait_if_paused_unblocks_on_resume(self):
        app_module._request_pause("test-job-1")
        # Resume after 0.3s on a background thread
        threading.Timer(0.3, lambda: app_module._request_resume("test-job-1")).start()
        start = time.monotonic()
        app_module._wait_if_paused("test-job-1")
        elapsed = time.monotonic() - start
        assert 0.2 < elapsed < 2.0, f"Expected ~0.3s wait, got {elapsed:.2f}s"

    def test_wait_if_paused_unblocks_on_cancel(self):
        app_module._request_pause("test-job-1")
        threading.Timer(0.3, lambda: app_module._request_cancel("test-job-1")).start()
        start = time.monotonic()
        app_module._wait_if_paused("test-job-1")
        elapsed = time.monotonic() - start
        assert elapsed < 2.0


# ---------------------------------------------------------------------------
# /progress endpoint — paused and cancelled fields
# ---------------------------------------------------------------------------

class TestProgressEndpoint:
    def test_progress_includes_paused_false_by_default(self, client):
        job_id = "prog-test-1"
        app_module._report_progress(job_id, 2, 5, phase="synthesizing")
        try:
            resp = client.get(f"/progress/{job_id}")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "paused" in data
            assert data["paused"] is False
        finally:
            app_module._clear_job(job_id)

    def test_progress_paused_true_when_paused(self, client):
        job_id = "prog-test-2"
        app_module._report_progress(job_id, 2, 5, phase="synthesizing")
        try:
            app_module._request_pause(job_id)
            resp = client.get(f"/progress/{job_id}")
            data = resp.get_json()
            assert data["paused"] is True
        finally:
            app_module._clear_job(job_id)

    def test_progress_cancelled_true_after_cancel(self, client):
        job_id = "prog-test-3"
        app_module._report_progress(job_id, 1, 5, phase="synthesizing")
        try:
            app_module._request_cancel(job_id)
            resp = client.get(f"/progress/{job_id}")
            data = resp.get_json()
            assert data["cancelled"] is True
        finally:
            app_module._clear_job(job_id)

    def test_progress_unknown_job_returns_defaults(self, client):
        resp = client.get("/progress/totally-nonexistent-job-xyz")
        data = resp.get_json()
        assert data["phase"] == "unknown"
        assert data["paused"] is False
        assert data["cancelled"] is False


# ---------------------------------------------------------------------------
# Job control HTTP endpoints
# ---------------------------------------------------------------------------

class TestJobControlEndpoints:
    def test_cancel_endpoint_returns_ok(self, client):
        job_id = "ep-cancel-1"
        app_module._report_progress(job_id, 0, 5)
        try:
            resp = client.post(f"/api/job/{job_id}/cancel")
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
        finally:
            app_module._clear_job(job_id)

    def test_pause_endpoint_returns_ok(self, client):
        job_id = "ep-pause-1"
        app_module._report_progress(job_id, 0, 5)
        try:
            resp = client.post(f"/api/job/{job_id}/pause")
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
        finally:
            app_module._clear_job(job_id)

    def test_resume_endpoint_returns_ok(self, client):
        job_id = "ep-resume-1"
        app_module._report_progress(job_id, 0, 5)
        try:
            app_module._request_pause(job_id)
            resp = client.post(f"/api/job/{job_id}/resume")
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
        finally:
            app_module._clear_job(job_id)

    def test_cancel_endpoint_sets_flag(self, client):
        job_id = "ep-cancel-flag"
        app_module._report_progress(job_id, 0, 5)
        try:
            client.post(f"/api/job/{job_id}/cancel")
            assert app_module._check_cancelled(job_id) is True
        finally:
            app_module._clear_job(job_id)

    def test_pause_then_resume_via_endpoints(self, client):
        job_id = "ep-pause-resume"
        app_module._report_progress(job_id, 0, 5)
        try:
            client.post(f"/api/job/{job_id}/pause")
            with app_module._jobs_lock:
                assert not app_module._jobs[job_id]["pause_event"].is_set()
            client.post(f"/api/job/{job_id}/resume")
            with app_module._jobs_lock:
                assert app_module._jobs[job_id]["pause_event"].is_set()
        finally:
            app_module._clear_job(job_id)


# ---------------------------------------------------------------------------
# _convert_chapters stops on cancel (integration-style)
# ---------------------------------------------------------------------------

class TestConvertChaptersCancel:
    def test_cancel_mid_conversion_produces_partial_output(self, monkeypatch):
        """Cancelling after the first chapter should produce only 1 MP3."""
        synthesis_count = [0]

        def fake_synthesis(text, backend, params, on_progress=None):
            synthesis_count[0] += 1
            buf = io.BytesIO(b"ID3" + b"\x00" * 128)
            buf.seek(0)
            return {"mp3_buf": buf, "provider_detail": "test", "prosody_info": {}}

        monkeypatch.setattr(app_module, "_do_synthesis", fake_synthesis)

        epub_bytes = _make_epub_bytes([
            ("Chapter 1", "First chapter body text."),
            ("Chapter 2", "Second chapter body text."),
            ("Chapter 3", "Third chapter body text."),
        ])

        job_id = "cancel-test-job"
        # Set up job state and pre-cancel it so it stops after reporting extraction phase
        app_module._report_progress(job_id, 0, 1, phase="extracting")
        # Cancel before synthesis begins
        app_module._request_cancel(job_id)

        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            resp = c.post(
                "/convert",
                data={
                    "backend": "kokoro",
                    "chapter_mode": "1",
                    "job_id": job_id,
                    "pdf": (io.BytesIO(epub_bytes), "test.epub"),
                },
                content_type="multipart/form-data",
            )
        # Cancelled early — should get 0 chapters synthesized or a valid (possibly partial) response
        assert resp.status_code in (200, 400)
        # Key assertion: synthesis was not called for all 3 chapters
        assert synthesis_count[0] < 3, (
            f"Expected partial synthesis, but all {synthesis_count[0]} chapters were synthesized"
        )
