import logging
from pathlib import Path
import pypdf as PyPDF2  # Keep same API
import pdfplumber

class PDFProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF (main method for Week 1)"""
        # Try pdfplumber first (more reliable)
        text = self._extract_with_pdfplumber(pdf_path)
        
        if not text.strip():
            # Fallback to PyPDF2
            text = self._extract_with_pypdf2(pdf_path)

        if not text.strip():
            # OCR fallback for scanned PDFs
            text = self._extract_with_ocr(pdf_path)

        return self._clean_text(text) if text else ""
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed for {pdf_path.name}: {e}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract using PyPDF2 fallback"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return ""

    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """OCR extraction for scanned PDFs."""
        try:
            import pytesseract
            from pdf2image import convert_from_path

            images = convert_from_path(pdf_path)
            text = ""

            for image in images:
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"

            cleaned = text.strip()
            self.logger.info(f"OCR extracted {len(cleaned)} chars from {pdf_path.name}")
            return cleaned
        except Exception as e:
            self.logger.error(f"OCR failed for {pdf_path.name}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        import re
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Normalize spaces/tabs but preserve newlines for regex-based question parsing.
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Collapse repeated blank lines.
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
