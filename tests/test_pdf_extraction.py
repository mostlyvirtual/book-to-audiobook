"""Tests for Phase 1: PyMuPDF-based PDF extraction, spatial filtering, OCR fallback."""
import io
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# Make sure the project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
import app as app_module


# ---------------------------------------------------------------------------
# Helpers — minimal in-memory PDF via fitz
# ---------------------------------------------------------------------------

def _make_fitz_doc_with_text(pages: list[str]):
    """Build an in-memory fitz document with one text block per page."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((72, 100), text, fontsize=12)
    return doc


def _make_fitz_doc_stream(pages: list[str]) -> io.BytesIO:
    """Return a BytesIO of a minimal PDF with the given per-page texts."""
    import fitz

    doc = _make_fitz_doc_with_text(pages)
    buf = io.BytesIO(doc.write())
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Phase 1.1 — basic fitz text extraction
# ---------------------------------------------------------------------------

class TestFitzBasicExtraction:
    def test_extracts_text_from_single_page(self):
        stream = _make_fitz_doc_stream(["Hello world. This is a test page."])
        text, label = app_module.extract_text_from_pdf(stream, "all")
        assert "Hello world" in text
        assert label == "all"

    def test_extracts_text_from_page_range(self):
        stream = _make_fitz_doc_stream(["Page one text.", "Page two text.", "Page three text."])
        text, label = app_module.extract_text_from_pdf(stream, "1-2")
        assert "Page one" in text
        assert "Page two" in text
        assert "Page three" not in text
        assert label == "p1-2"

    def test_single_page_label(self):
        stream = _make_fitz_doc_stream(["Only page.", "Second page."])
        _, label = app_module.extract_text_from_pdf(stream, "2")
        assert label == "p2"

    def test_invalid_page_range_raises(self):
        stream = _make_fitz_doc_stream(["Single page."])
        with pytest.raises(ValueError, match="exceeds"):
            app_module.extract_text_from_pdf(stream, "5")


# ---------------------------------------------------------------------------
# Phase 1.2 — spatial header/footer filtering in _extract_page_text_fitz
# ---------------------------------------------------------------------------

class TestSpatialFiltering:
    def test_block_in_header_zone_is_excluded(self):
        """A block with y1 < header_threshold should be dropped."""
        import fitz

        page = mock.MagicMock()
        page.rect.height = 842
        # Block entirely in the header zone (y1 = 30 < default threshold 50)
        blocks = [
            (10, 10, 200, 30, "RUNNING HEADER", 0, 0),  # in header zone
            (10, 100, 500, 200, "Body text here.", 1, 0),  # normal body
        ]
        # _extract_page_text_fitz calls get_text("blocks") for spatial filtering
        # and _is_scanned_page calls get_text() (no args) expecting a string.
        page.get_text.side_effect = lambda fmt=None, **kw: blocks if fmt == "blocks" else "RUNNING HEADER\nBody text here."
        page.get_images.return_value = []

        result = app_module._extract_page_text_fitz(page, header_threshold=50)
        assert "RUNNING HEADER" not in result
        assert "Body text here" in result

    def test_block_in_footer_zone_is_excluded(self):
        """A block starting below page_height - footer_threshold should be dropped."""
        import fitz

        page = mock.MagicMock()
        page.rect.height = 842
        # Block in footer zone (y0 = 800 > 842 - 50 = 792)
        blocks = [
            (10, 100, 500, 200, "Body text here.", 0, 0),
            (10, 800, 500, 840, "12", 1, 0),  # footer page number
        ]
        page.get_text.side_effect = lambda fmt=None, **kw: blocks if fmt == "blocks" else "Body text here.\n12"
        page.get_images.return_value = []

        result = app_module._extract_page_text_fitz(page, footer_threshold=50)
        assert "Body text here" in result
        assert "12" not in result

    def test_normal_body_block_is_included(self):
        page = mock.MagicMock()
        page.rect.height = 842
        blocks = [(10, 70, 500, 400, "Normal content.", 0, 0)]
        page.get_text.side_effect = lambda fmt=None, **kw: blocks if fmt == "blocks" else "Normal content."
        page.get_images.return_value = []

        result = app_module._extract_page_text_fitz(page)
        assert "Normal content" in result


# ---------------------------------------------------------------------------
# Phase 1.3 — scanned page detection
# ---------------------------------------------------------------------------

class TestScannedPageDetection:
    def test_page_with_text_is_not_scanned(self):
        page = mock.MagicMock()
        page.get_text.return_value = "This is a normal text page with plenty of content."
        page.get_images.return_value = []
        assert app_module._is_scanned_page(page) is False

    def test_page_with_images_and_no_text_is_scanned(self):
        page = mock.MagicMock()
        page.get_text.return_value = "   "  # < 50 chars
        page.get_images.return_value = [("img1",)]
        assert app_module._is_scanned_page(page) is True

    def test_page_with_images_but_text_is_not_scanned(self):
        page = mock.MagicMock()
        page.get_text.return_value = "There is readable text here plus some more words to exceed the threshold."
        page.get_images.return_value = [("img1",)]
        assert app_module._is_scanned_page(page) is False

    def test_empty_page_without_images_is_not_scanned(self):
        page = mock.MagicMock()
        page.get_text.return_value = ""
        page.get_images.return_value = []
        assert app_module._is_scanned_page(page) is False


# ---------------------------------------------------------------------------
# Phase 1.4 — OCR fallback triggered for scanned pages
# ---------------------------------------------------------------------------

class TestOCRFallback:
    def test_ocr_called_for_scanned_page(self, monkeypatch):
        """_extract_page_text_fitz should call _ocr_page when page is scanned."""
        calls = []

        def fake_ocr(page, dpi=300):
            calls.append(True)
            return "OCR extracted text"

        monkeypatch.setattr(app_module, "_ocr_page", fake_ocr)

        page = mock.MagicMock()
        page.rect.height = 842
        # get_text("blocks") returns little text, triggering OCR check
        page.get_text.side_effect = lambda fmt=None, **kw: (
            [] if fmt == "blocks" else "  "  # bare string for _is_scanned_page check
        )
        page.get_images.return_value = [("img",)]

        result = app_module._extract_page_text_fitz(page)
        assert calls, "OCR was not called for a scanned page"
        assert "OCR extracted text" in result

    def test_ocr_not_called_for_text_page(self, monkeypatch):
        """_extract_page_text_fitz must NOT call OCR when the page has real text."""
        calls = []
        monkeypatch.setattr(app_module, "_ocr_page", lambda p, **kw: calls.append(True) or "")

        page = mock.MagicMock()
        page.rect.height = 842
        blocks = [(10, 100, 500, 300, "Plenty of readable body text on this page.", 0, 0)]
        # Both get_text("blocks") and get_text() (no args) must return the right types.
        page.get_text.side_effect = lambda fmt=None, **kw: blocks if fmt == "blocks" else "Plenty of readable body text on this page."
        page.get_images.return_value = []

        app_module._extract_page_text_fitz(page)
        assert not calls, "OCR was called unnecessarily on a text page"


# ---------------------------------------------------------------------------
# Phase 1.5 — NFKC normalization in preprocess_text_for_speech
# ---------------------------------------------------------------------------

class TestNFKCNormalization:
    def test_ligature_fi_normalized(self):
        # U+FB01 ﬁ (fi ligature) should become "fi" after NFKC
        text = "\ufb01ne day"
        result = app_module.preprocess_text_for_speech(text)
        assert "ﬁ" not in result
        assert "fi" in result

    def test_ligature_fl_normalized(self):
        text = "\ufb02oor"
        result = app_module.preprocess_text_for_speech(text)
        assert "fl" in result

    def test_fullwidth_chars_normalized(self):
        # Full-width letters (U+FF21 etc.) should map to ASCII
        text = "\uff28\uff25\uff2c\uff2c\uff2f"  # HELLO in full-width
        result = app_module.preprocess_text_for_speech(text)
        assert "HELLO" in result or "Hello" in result

    def test_normal_text_unchanged_by_nfkc(self):
        text = "The quick brown fox."
        result = app_module.preprocess_text_for_speech(text)
        assert "The quick brown fox" in result
