import os
import time
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.extractor import extract_text
from services.nlp import analyze_text


class DocumentRequest(BaseModel):
    fileName: str = Field(..., min_length=1)
    fileType: str = Field(..., pattern="^(pdf|docx|png|jpg|jpeg)$")
    fileBase64: str = Field(..., min_length=1)


class EntitiesResponse(BaseModel):
    PERSON: list[str]
    ORG: list[str]
    DATE: list[str]
    MONEY: list[str]
    GPE: list[str]


class DocumentResponse(BaseModel):
    summary: str
    entities: EntitiesResponse
    sentiment: str = Field(..., pattern="^(positive|negative|neutral)$")
    type: str = Field(..., pattern="^(invoice|resume|legal|general)$")
    sensitive_data: dict
    confidence: dict
    processing_time: str


def _get_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is not set")
    return api_key


def _is_valid_bearer_token(authorization_header: str, expected_api_key: str) -> bool:
    parts = (authorization_header or "").split()
    if len(parts) != 2:
        return False
    if parts[0].lower() != "bearer":
        return False
    return parts[1] == expected_api_key


def api_key_auth(request: Request) -> None:
    expected = _get_api_key()

    auth = request.headers.get("Authorization", "")
    if not _is_valid_bearer_token(auth, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


app = FastAPI(title="AI-Powered Document Analysis & Extraction API")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, __: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=DocumentResponse, dependencies=[Depends(api_key_auth)])
def analyze(doc: DocumentRequest):
    start = time.perf_counter()
    text = extract_text(
        file_base64=doc.fileBase64,
        file_type=doc.fileType,
        file_name=doc.fileName,
    )

    result = analyze_text(text)

    elapsed = time.perf_counter() - start

    return {
        "summary": result["summary"],
        "entities": result["entities"],
        "sentiment": result["sentiment"],
        "type": result.get("type", "general"),
        "sensitive_data": result.get("sensitive_data", {"has_sensitive": False, "types": []}),
        "confidence": result.get("confidence", {"entities": 0.0, "sentiment": 0.0}),
        "processing_time": f"{elapsed:.2f}s",
    }
