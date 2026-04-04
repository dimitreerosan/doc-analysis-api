import os
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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


def _get_api_key() -> str:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY environment variable is not set")
    return api_key


security = HTTPBearer()


def api_key_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    expected = _get_api_key()

    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


app = FastAPI(title="AI-Powered Document Analysis & Extraction API")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/")
def root():
    return {
        "status": "running",
        "project": "AI-Powered Document Analysis & Extraction API",
        "creator": "DIMITREE",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze  (Bearer token required)",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=DocumentResponse, dependencies=[Depends(api_key_auth)])
def analyze(doc: DocumentRequest):
    text = extract_text(
        file_base64=doc.fileBase64,
        file_type=doc.fileType,
        file_name=doc.fileName,
    )

    result = analyze_text(text)

    return {
        "summary": result["summary"],
        "entities": result["entities"],
        "sentiment": result["sentiment"],
    }
