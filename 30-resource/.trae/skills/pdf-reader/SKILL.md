---
name: "pdf-reader"
description: "Reads and extracts text from PDF files using OCR when needed. Invoke when user asks to read/analyze PDF documents or extract content from PDF files."
---

# PDF Reader

This skill reads and extracts text content from PDF files, handling both native text PDFs and image-based/scanned PDFs using OCR.

## Capabilities

1. **Native Text Extraction**: Extract text directly from PDFs with embedded text
2. **OCR Processing**: When text extraction fails (scanned PDFs), render pages as images and use OCR
3. **Full Content Reading**: Read entire PDF or specific page ranges
4. **Content Analysis**: Provide structured summary of document content

## Usage

### Basic Text Extraction
```python
import fitz  # PyMuPDF
doc = fitz.open("document.pdf")
for page in doc:
    text = page.get_text()
    print(text)
```

### OCR for Scanned PDFs
```python
import fitz
from PIL import Image
import pytesseract

doc = fitz.open("scanned.pdf")
for page_num, page in enumerate(doc):
    mat = fitz.Matrix(2, 2)  # 2x resolution for better OCR
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.tobytes("png"))
    text = pytesseract.image_to_string(img, lang='chi_sim+eng')
    print(f"Page {page_num + 1}: {text}")
```

### Handling Large PDFs
- Process in batches to avoid memory issues
- Save extracted text to intermediate files
- Use concurrent processing for faster extraction

## Prerequisites

- PyMuPDF (`pip install pymupdf`)
- Pillow (`pip install pillow`)
- Tesseract OCR (system installation)
- Chinese language data for tesseract (`tesseract-ocr-chi-sim`)

## Error Handling

1. If `get_text()` returns empty, try OCR approach
2. If OCR fails, report that PDF is image-based and cannot be processed
3. Handle corrupted PDF gracefully with error message