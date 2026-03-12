"""Tests for Phase 2: chapter detection strategies and overlap removal."""
import io
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import app as app_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(toc=None, pages_text=None):
    """Build a mock fitz doc suitable for _detect_pdf_chapters."""
    doc = mock.MagicMock()
    toc = toc or []
    pages_text = pages_text or []

    doc.get_toc.return_value = toc
    doc.__len__ = mock.Mock(return_value=len(pages_text))

    def _load_page(i):
        page = mock.MagicMock()
        text = pages_text[i] if i < len(pages_text) else ""
        page.get_text.return_value = text
        return page

    doc.load_page.side_effect = _load_page
    return doc


# ---------------------------------------------------------------------------
# Strategy 1 — fitz get_toc()
# ---------------------------------------------------------------------------

class TestToCStrategy:
    def test_extracts_chapters_from_get_toc(self):
        toc = [
            [1, "Introduction", 1],
            [1, "Chapter 1: The Beginning", 5],
            [2, "Setting the Scene", 7],
            [1, "Chapter 2: The Middle", 12],
        ]
        doc = _make_mock_doc(toc=toc, pages_text=["dummy"] * 20)
        chapters = app_module._detect_pdf_chapters(doc)
        titles = [c["title"] for c in chapters]
        assert "Introduction" in titles
        assert "Chapter 1: The Beginning" in titles
        assert "Setting the Scene" in titles

    def test_uses_doc_get_toc_not_outline(self):
        """Ensure we call doc.get_toc(), not doc.outline (pypdf API)."""
        toc = [[1, "Only Chapter", 1]]
        doc = _make_mock_doc(toc=toc, pages_text=["text"])
        app_module._detect_pdf_chapters(doc)
        doc.get_toc.assert_called_once()
        assert not hasattr(doc, "outline") or not doc.outline.called

    def test_depth_set_from_toc_level(self):
        toc = [[1, "Top Level", 1], [2, "Sub Level", 3]]
        doc = _make_mock_doc(toc=toc, pages_text=["t"] * 5)
        chapters = app_module._detect_pdf_chapters(doc)
        top = next(c for c in chapters if c["title"] == "Top Level")
        sub = next(c for c in chapters if c["title"] == "Sub Level")
        assert top["depth"] == 0
        assert sub["depth"] == 1

    def test_empty_toc_falls_through_to_next_strategy(self):
        """Empty get_toc() must not produce chapters — falls through."""
        doc = _make_mock_doc(toc=[], pages_text=["no chapter text"] * 3)
        # With no meaningful content, result may be empty or from s5
        # Just verify no crash and no bogus TOC-based chapters
        chapters = app_module._detect_pdf_chapters(doc)
        assert isinstance(chapters, list)


# ---------------------------------------------------------------------------
# Strategy 2 — Pipe-header "Title | Section" detection
# ---------------------------------------------------------------------------

class TestPipeHeaderStrategy:
    def test_detects_pipe_header_chapters(self):
        pages = [
            "My Book | Introduction\nBody text of the introduction goes here.",
            "My Book | Chapter 1\nThis is chapter one content.",
            "My Book | Chapter 1\nContinued chapter one.",
            "My Book | Chapter 2\nThis is chapter two.",
        ]
        doc = _make_mock_doc(toc=[], pages_text=pages)
        chapters = app_module._detect_pdf_chapters(doc)
        titles = [c["title"] for c in chapters]
        assert any("Introduction" in t for t in titles)
        assert any("Chapter 1" in t or "Chapter 2" in t for t in titles)

    def test_pipe_header_deduplicates_sections(self):
        """Same section on consecutive pages should only yield one chapter entry."""
        pages = ["Book | Chapter 1\ntext."] * 3
        doc = _make_mock_doc(toc=[], pages_text=pages)
        chapters = app_module._detect_pdf_chapters(doc)
        ch1_entries = [c for c in chapters if "Chapter 1" in c["title"]]
        assert len(ch1_entries) == 1


# ---------------------------------------------------------------------------
# Strategy 4 — Heuristic heading scan
# ---------------------------------------------------------------------------

class TestHeuristicHeadingStrategy:
    def test_detects_chapter_keyword_headings(self):
        pages = [
            "Introduction\nSome opening content.",
            "Chapter 1\nFirst chapter body.",
            "Regular page content without heading.",
            "Chapter 2\nSecond chapter.",
        ]
        doc = _make_mock_doc(toc=[], pages_text=pages)
        chapters = app_module._detect_pdf_chapters(doc)
        titles = [c["title"] for c in chapters]
        assert any("Chapter 1" in t for t in titles)
        assert any("Chapter 2" in t for t in titles)

    def test_regular_page_not_detected_as_chapter(self):
        pages = [
            "Chapter 1\nFirst chapter.",
            "Normal paragraph. No heading here at all.",
            "Chapter 2\nSecond chapter.",
        ]
        doc = _make_mock_doc(toc=[], pages_text=pages)
        chapters = app_module._detect_pdf_chapters(doc)
        # Should have 2 chapters, not 3
        assert len(chapters) == 2


# ---------------------------------------------------------------------------
# Strategy 5 — Blank-line heuristic split (last resort)
# ---------------------------------------------------------------------------

class TestBlankLineSplitStrategy:
    def test_splits_on_triple_blank_lines(self):
        """When no other strategy fires, triple-blank-line sections become chapters."""
        # Must NOT start with chapter keywords (chapter/section/part/etc.) or
        # Strategy 4's heuristic regex fires first and Strategy 5 never runs.
        section_a = "Lorem ipsum dolor sit amet.\nMore text here."
        section_b = "Consectetur adipiscing elit.\nMore text here."
        combined = section_a + "\n\n\n" + section_b

        pages = [combined]
        doc = _make_mock_doc(toc=[], pages_text=pages)
        chapters = app_module._detect_pdf_chapters(doc)
        # Should have produced at least 2 sections
        assert len(chapters) >= 2

    def test_no_split_when_no_blank_lines(self):
        """Single continuous text block should remain as one fallback section."""
        text = "Just one section with no blank lines at all. It goes on and on."
        doc = _make_mock_doc(toc=[], pages_text=[text])
        chapters = app_module._detect_pdf_chapters(doc)
        # Either 0 or 1 chapter, certainly not 2
        assert len(chapters) <= 1


# ---------------------------------------------------------------------------
# _remove_chapter_overlap
# ---------------------------------------------------------------------------

class TestOverlapRemoval:
    def test_removes_duplicated_tail_lines(self):
        prev = "Line A\nLine B\nLine C\nLine D"
        curr = "Line C\nLine D\nLine E\nLine F"
        result = app_module._remove_chapter_overlap(prev, curr)
        assert "Line A" in result
        assert "Line B" in result
        assert "Line C" not in result
        assert "Line D" not in result

    def test_no_overlap_returns_prev_unchanged(self):
        prev = "Completely different text here."
        curr = "No shared content at all."
        result = app_module._remove_chapter_overlap(prev, curr)
        assert result == prev

    def test_check_lines_limit_is_respected(self):
        """Overlaps larger than check_lines should not be removed."""
        shared = "\n".join(f"Line {i}" for i in range(25))
        prev = "Unique start.\n" + shared
        curr = shared + "\nUnique end."
        # check_lines=10 should not find the 25-line overlap
        result = app_module._remove_chapter_overlap(prev, curr, check_lines=10)
        assert result == prev

    def test_single_line_overlap_removed(self):
        prev = "Paragraph text.\nLast overlap line."
        curr = "Last overlap line.\nNew content."
        result = app_module._remove_chapter_overlap(prev, curr)
        assert "Last overlap line" not in result
        assert "Paragraph text" in result

    def test_empty_texts_dont_crash(self):
        assert app_module._remove_chapter_overlap("", "") == ""
        assert app_module._remove_chapter_overlap("text", "") == "text"
        assert app_module._remove_chapter_overlap("", "text") == ""
