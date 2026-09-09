"""Unit tests for the Project Gutenberg data preparation pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from scripts.prepare_gutenberg import (
    DOC_SEP,
    clean_text,
    decode_bytes,
    is_duplicate,
    is_html_or_corrupt,
    strip_gutenberg_boilerplate,
    build_corpus_from_files,
)


class TestBoilerplateStripping:
    def test_standard_markers(self):
        raw = (
            "The Project Gutenberg EBook of Frankenstein, by Mary Shelley\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***\n"
            "Letter 1\nTo Mrs. Saville, England.\nSt. Petersburgh, Dec. 11th, 17--.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK FRANKENSTEIN ***\n"
            "End of Project Gutenberg's Frankenstein\n"
        )
        cleaned, has_markers = strip_gutenberg_boilerplate(raw)
        assert has_markers is True
        assert "Letter 1" in cleaned
        assert "START OF" not in cleaned
        assert "END OF" not in cleaned
        assert "Mrs. Saville" in cleaned

    def test_this_ebook_variant(self):
        raw = (
            "Some header information\n"
            "*** START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
            "It is a truth universally acknowledged, that a single man in possession...\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
            "Legal license text"
        )
        cleaned, has_markers = strip_gutenberg_boilerplate(raw)
        assert has_markers is True
        assert cleaned.startswith("It is a truth universally acknowledged")
        assert "START OF" not in cleaned
        assert "Legal license text" not in cleaned

    def test_no_markers_fallback(self):
        raw = "Just plain text without any Gutenberg markers at all."
        cleaned, has_markers = strip_gutenberg_boilerplate(raw)
        assert has_markers is False
        assert cleaned == raw


class TestHtmlAndCorruptionDetection:
    def test_short_content_is_rejected(self):
        assert is_html_or_corrupt("Short text") is True

    def test_html_error_page_is_rejected(self):
        html_page = "<!DOCTYPE html><html><head><title>403 Forbidden</title></head><body><h1>Access Denied</h1></body></html>" * 30
        assert is_html_or_corrupt(html_page) is True

    def test_valid_book_content_is_accepted(self):
        valid_book = "Chapter 1. Loomings. Call me Ishmael. Some years ago--never mind how long precisely... " * 100
        assert is_html_or_corrupt(valid_book) is False


class TestTextCleaning:
    def test_crlf_normalization(self):
        raw = "Line 1\r\nLine 2\r\nLine 3"
        cleaned = clean_text(raw)
        assert "\r" not in cleaned
        assert cleaned == "Line 1\nLine 2\nLine 3"

    def test_excess_blank_lines(self):
        raw = "Paragraph 1\n\n\n\n\nParagraph 2"
        cleaned = clean_text(raw)
        assert cleaned == "Paragraph 1\n\nParagraph 2"

    def test_horizontal_spaces(self):
        raw = "Word1    Word2\t\tWord3"
        cleaned = clean_text(raw)
        assert cleaned == "Word1 Word2 Word3"

    def test_hyphen_rejoining(self):
        raw = "This is a demon-\nstration of hyphen re-\njoining across lines."
        cleaned = clean_text(raw)
        assert "demonstration" in cleaned
        assert "rejoining" in cleaned


class TestEncodingHandling:
    def test_utf8_decoding(self):
        data = "Bonjour, café et thé!".encode("utf-8")
        assert decode_bytes(data) == "Bonjour, café et thé!"

    def test_latin1_fallback(self):
        # Latin-1 byte for accented é is 0xE9
        data = b"Caf\xe9 au lait"
        decoded = decode_bytes(data)
        assert "Café au lait" in decoded


class TestDeduplication:
    def test_duplicate_detection(self):
        seen = set()
        text_a = "It was the best of times, it was the worst of times... " * 50
        text_b = text_a  # exact duplicate
        text_c = "Alice was beginning to get very tired of sitting by her sister... " * 50

        assert is_duplicate(text_a, seen) is False
        assert is_duplicate(text_b, seen) is True
        assert is_duplicate(text_c, seen) is False


class TestCorpusBuilder:
    def test_build_corpus_with_doc_separator(self, tmp_path):
        doc1_path = tmp_path / "pg1.txt"
        doc2_path = tmp_path / "pg2.txt"

        book1_text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK BOOK ONE ***\n"
            + "First book story content line. " * 100 + "\n"
            + "*** END OF THE PROJECT GUTENBERG EBOOK BOOK ONE ***"
        )
        book2_text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK BOOK TWO ***\n"
            + "Second book story content line. " * 100 + "\n"
            + "*** END OF THE PROJECT GUTENBERG EBOOK BOOK TWO ***"
        )

        doc1_path.write_text(book1_text, encoding="utf-8")
        doc2_path.write_text(book2_text, encoding="utf-8")

        out_corpus = tmp_path / "corpus.txt"
        docs, chars = build_corpus_from_files([doc1_path, doc2_path], out_corpus)

        assert docs == 2
        assert chars > 0
        content = out_corpus.read_text(encoding="utf-8")
        assert DOC_SEP in content
        assert "First book story content line." in content
        assert "Second book story content line." in content
        assert content.count(DOC_SEP) == 2
