"""Unit tests for PDFService using synthetic test PDFs created with PyMuPDF."""
from pathlib import Path
import pytest
import pymupdf

from app.services.pdf_service import (
    PDFService,
    PDFFileNotFoundError,
    InvalidPDFError,
    PDFPageLimitExceededError,
    PDFExtractionError,
)


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper utility to dynamically generate a test PDF file with specified page contents."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            # Insert text at standard margins (x=50, y=72)
            page.insert_text((50, 72), text)
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture
def pdf_service() -> PDFService:
    """Fixture providing a fresh PDFService instance."""
    return PDFService()


class TestPDFServiceSuccess:
    """Tests for successful PDF extraction scenarios."""

    def test_extract_single_page_pdf(self, tmp_path: Path, pdf_service: PDFService):
        """Verifies extracting text from a single-page PDF document."""
        pdf_path = create_sample_pdf(tmp_path / "single.pdf", ["This is page one content."])

        result = pdf_service.extract_text(pdf_path)

        assert result.total_pages == 1
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1
        assert "This is page one content." in result.pages[0].text
        assert not result.pages[0].is_empty
        assert result.pages[0].char_count > 0

    def test_extract_multi_page_pdf(self, tmp_path: Path, pdf_service: PDFService):
        """Verifies extracting text and 1-based page numbering across multiple pages."""
        pages_content = [
            "Introduction to ResearchLens AI.",
            "Methodology and system design.",
            "Results, discussion, and conclusions.",
        ]
        pdf_path = create_sample_pdf(tmp_path / "multi.pdf", pages_content)

        result = pdf_service.extract_text(pdf_path)

        assert result.total_pages == 3
        assert len(result.pages) == 3

        # Verify exact 1-based page numbering sequence
        for idx, page in enumerate(result.pages):
            assert page.page_number == idx + 1
            assert pages_content[idx] in page.text

        # Verify full_text property
        assert "Introduction to ResearchLens AI." in result.full_text
        assert "Results, discussion, and conclusions." in result.full_text

    def test_preserve_page_numbering_with_empty_and_scanned_pages(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that blank pages preserve their exact 1-based page number without shifting."""
        pages_content = [
            "Page 1: Title and Abstract",
            "",  # Completely empty page (simulates blank page or scanned page without OCR)
            "Page 3: Body and Methodology",
        ]
        pdf_path = create_sample_pdf(tmp_path / "sparse.pdf", pages_content)

        result = pdf_service.extract_text(pdf_path)

        assert result.total_pages == 3
        assert len(result.pages) == 3

        # Page 1
        assert result.pages[0].page_number == 1
        assert "Page 1: Title and Abstract" in result.pages[0].text
        assert not result.pages[0].is_empty

        # Page 2: must exist in sequence, with page_number == 2 and empty text
        assert result.pages[1].page_number == 2
        assert result.pages[1].text == ""
        assert result.pages[1].is_empty
        assert result.pages[1].char_count == 0

        # Page 3: must be page_number == 3, NOT shifted to 2
        assert result.pages[2].page_number == 3
        assert "Page 3: Body and Methodology" in result.pages[2].text
        assert not result.pages[2].is_empty


class TestPDFServiceErrors:
    """Tests for error handling, corrupt inputs, and invalid files."""

    def test_missing_file_raises_pdf_file_not_found(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that non-existent paths raise PDFFileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFFileNotFoundError) as exc_info:
            pdf_service.extract_text(non_existent)
        assert "does not exist" in str(exc_info.value)

    def test_directory_path_raises_pdf_file_not_found(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that passing a directory path raises PDFFileNotFoundError."""
        with pytest.raises(PDFFileNotFoundError) as exc_info:
            pdf_service.extract_text(tmp_path)
        assert "is not a file" in str(exc_info.value)

    def test_empty_zero_byte_file_raises_invalid_pdf(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that a 0-byte file raises InvalidPDFError."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.touch()

        with pytest.raises(InvalidPDFError) as exc_info:
            pdf_service.extract_text(empty_file)
        assert "empty (0 bytes)" in str(exc_info.value)

    def test_non_pdf_file_raises_invalid_pdf(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that a plain text or non-PDF file without magic bytes raises InvalidPDFError."""
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("This is just a plain text file, not a PDF document.")

        with pytest.raises(InvalidPDFError) as exc_info:
            pdf_service.extract_text(fake_pdf)
        assert "magic bytes" in str(exc_info.value)

    def test_corrupted_pdf_content_raises_invalid_pdf(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that a file with PDF magic bytes but broken internal structure raises InvalidPDFError."""
        corrupted_file = tmp_path / "corrupted.pdf"
        # Magic bytes '%PDF-1.7' followed by garbled binary junk that fails parser
        corrupted_file.write_bytes(b"%PDF-1.7\n\x00\xff\xfe\x01\x02\x03CORRUPTED_BYTES_NO_XREF_EOF")

        with pytest.raises(InvalidPDFError) as exc_info:
            pdf_service.extract_text(corrupted_file)
        assert "PyMuPDF could not parse" in str(exc_info.value) or "magic bytes" in str(exc_info.value)

    def test_extract_within_max_pages_limit(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that PDF with page count <= max_pages extracts successfully."""
        pdf_path = create_sample_pdf(tmp_path / "three_pages.pdf", ["Page 1", "Page 2", "Page 3"])
        result = pdf_service.extract_text(pdf_path, max_pages=3)
        assert result.total_pages == 3

    def test_extract_exceeding_max_pages_limit_raises_error(
        self, tmp_path: Path, pdf_service: PDFService
    ):
        """Verifies that PDF with page count > max_pages raises PDFPageLimitExceededError."""
        pdf_path = create_sample_pdf(tmp_path / "four_pages.pdf", ["Page 1", "Page 2", "Page 3", "Page 4"])
        with pytest.raises(PDFPageLimitExceededError) as exc_info:
            pdf_service.extract_text(pdf_path, max_pages=3)
        assert "exceeds maximum allowed limit of 3 pages" in str(exc_info.value)
