"""PDF Extraction Service for ResearchLens AI using PyMuPDF."""
from pathlib import Path
from typing import List, Union
import pymupdf
from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Represents text extracted from a single page of a PDF document."""
    page_number: int = Field(..., description="1-based physical page number")
    text: str = Field(..., description="Extracted plain text content")

    @property
    def char_count(self) -> int:
        """Returns the character count of extracted text."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """Indicates if the page contains no extractable text (e.g. blank or scanned)."""
        return len(self.text.strip()) == 0


class PDFExtractionResult(BaseModel):
    """Represents the complete text extraction result from a PDF document."""
    total_pages: int = Field(..., description="Total number of pages in the PDF")
    pages: List[ExtractedPage] = Field(..., description="Ordered list of extracted pages")

    @property
    def full_text(self) -> str:
        """Concatenates text across all pages separated by double newlines."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


# --- Custom Domain Exceptions ---

class PDFExtractionError(Exception):
    """Base exception for all PDF extraction failures."""
    pass


class PDFFileNotFoundError(PDFExtractionError):
    """Raised when the specified PDF file cannot be found or is not a file."""
    pass


class InvalidPDFError(PDFExtractionError):
    """Raised when the file is not a valid PDF or has corrupted structure."""
    pass


class EncryptedPDFError(PDFExtractionError):
    """Raised when the PDF is password-protected or encrypted."""
    pass


# --- Service Implementation ---

class PDFService:
    """Service responsible for extracting structured text and page metadata from PDFs."""

    PDF_MAGIC_BYTES = b"%PDF-"

    @classmethod
    def _validate_file_path(cls, file_path: Union[str, Path]) -> Path:
        """Validates that the file exists and is a readable regular file."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise PDFFileNotFoundError(f"PDF file does not exist or is not a file: {path}")

        # Check for minimum non-empty file size
        if path.stat().st_size == 0:
            raise InvalidPDFError(f"PDF file is empty (0 bytes): {path}")

        # Quick magic byte inspection before opening with C parser
        try:
            with open(path, "rb") as f:
                header = f.read(1024)
                if cls.PDF_MAGIC_BYTES not in header:
                    raise InvalidPDFError(
                        f"File does not contain valid PDF magic bytes ('%PDF-'): {path.name}"
                    )
        except (IOError, OSError) as e:
            raise PDFExtractionError(f"Failed to read file from disk: {e}") from e

        return path

    def extract_text(self, file_path: Union[str, Path]) -> PDFExtractionResult:
        """Extracts text page-by-page from the provided PDF file path.

        Args:
            file_path: Absolute or relative path to the target PDF file.

        Returns:
            PDFExtractionResult containing total page count and ordered ExtractedPage items.

        Raises:
            PDFFileNotFoundError: If the file does not exist.
            InvalidPDFError: If the file is not a valid PDF, has no pages, or is corrupt.
            EncryptedPDFError: If the PDF is encrypted and requires a password.
            PDFExtractionError: For unexpected extraction failures.
        """
        validated_path = self._validate_file_path(file_path)

        try:
            # Using context manager guarantees file handle is closed upon exit
            with pymupdf.open(validated_path) as doc:
                if doc.is_encrypted and doc.needs_pass:
                    raise EncryptedPDFError(
                        f"PDF is encrypted and password-protected: {validated_path.name}"
                    )

                if doc.page_count == 0:
                    raise InvalidPDFError(f"PDF contains 0 pages: {validated_path.name}")

                pages: List[ExtractedPage] = []
                for page_idx in range(doc.page_count):
                    page = doc.load_page(page_idx)
                    # Extract plain text; strip leading/trailing whitespace
                    raw_text = page.get_text("text")
                    cleaned_text = raw_text.strip() if raw_text else ""

                    pages.append(
                        ExtractedPage(
                            page_number=page_idx + 1,  # 1-based page numbering
                            text=cleaned_text,
                        )
                    )

                return PDFExtractionResult(
                    total_pages=doc.page_count,
                    pages=pages,
                )

        except (PDFExtractionError, PDFFileNotFoundError, InvalidPDFError, EncryptedPDFError):
            # Re-raise known domain exceptions directly
            raise
        except pymupdf.FileDataError as e:
            raise InvalidPDFError(f"PyMuPDF could not parse file data (corrupted or malformed): {e}") from e
        except Exception as e:
            raise PDFExtractionError(f"Unexpected error during PDF extraction: {e}") from e
