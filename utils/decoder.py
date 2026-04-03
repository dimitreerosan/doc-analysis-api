import base64
import binascii

from fastapi import HTTPException, status


MAX_DECODED_BYTES = 10 * 1024 * 1024


def decode_base64_file(file_base64: str) -> bytes:
    if not file_base64 or not file_base64.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty fileBase64")

    normalized = file_base64.strip()

    if "," in normalized and normalized.lower().startswith("data:"):
        normalized = normalized.split(",", 1)[1]

    try:
        decoded = base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64")

    if not decoded:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Decoded file is empty")

    if len(decoded) > MAX_DECODED_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    return decoded
