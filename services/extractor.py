import io
import re

import fitz
import pytesseract
from docx import Document
from fastapi import HTTPException, status
from PIL import Image

from utils.decoder import decode_base64_file


_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _extract_pdf(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid PDF")

    parts: list[str] = []
    try:
        for page in doc:
            parts.append(page.get_text("text") or "")
    finally:
        doc.close()

    return "\n".join(parts)


def _extract_docx(data: bytes) -> str:
    try:
        bio = io.BytesIO(data)
        document = Document(bio)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid DOCX")

    parts: list[str] = []
    for p in document.paragraphs:
        if p.text:
            parts.append(p.text)

    return "\n".join(parts)


def _extract_image(data: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image")

    try:
        text = pytesseract.image_to_string(image)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR failed",
        )

    return text


def extract_text(file_base64: str, file_type: str, file_name: str) -> str:
    _ = file_name
    data = decode_base64_file(file_base64)

    if file_type == "pdf":
        text = _extract_pdf(data)
    elif file_type == "docx":
        text = _extract_docx(data)
    elif file_type in {"png", "jpg", "jpeg"}:
        text = _extract_image(data)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported fileType")

    cleaned = _clean_text(text)
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No text extracted")

    return cleaned
