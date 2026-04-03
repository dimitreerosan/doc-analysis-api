# AI-Powered Document Analysis & Extraction API (Backend)

FastAPI backend that accepts Base64-encoded documents (PDF, DOCX, images), extracts text, runs NLP (summary, entities, sentiment), and returns structured JSON.

## Tech stack

- FastAPI (API)
- PyMuPDF (PDF text extraction)
- python-docx (DOCX extraction)
- pytesseract + Pillow (OCR for images)
- spaCy `en_core_web_sm` (NER)
- VADER (sentiment)

## Project structure

- `main.py`
- `services/`
  - `extractor.py`
  - `nlp.py`
- `utils/`
  - `decoder.py`
- `requirements.txt`

## Setup (local)

### 1) Create venv and install deps

```bash
python -m venv .venv
# activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3) Install Tesseract (required for OCR)

- Install Tesseract OCR for your OS.
- Ensure `tesseract` is on PATH.

### 4) Configure API key

Set environment variable:

```bash
# Windows PowerShell
$env:API_KEY="Doc_Analysis_Extraction_KB_secure"
```

### 5) Run

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## API Authentication

Send header:

- `Authorization: Bearer <API_KEY>`

Example header:

```text
Authorization: Bearer Doc_Analysis_Extraction_KB_secure
```

Requests without a valid key get `401`.

## Endpoints

### `GET /health`

Returns:

```json
{ "status": "ok" }
```

Example (PowerShell) request:

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
$body = @{
  fileName = "sample.pdf"
  fileType = "pdf"
  fileBase64 = "<base64>"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:10000/analyze" -Headers $headers -Body $body -ContentType "application/json"
```

### `POST /analyze`

Request body:

```json
{
  "fileName": "sample.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64>"
}
```

Response:

```json
{
  "summary": "string",
  "entities": {
    "PERSON": [],
    "ORG": [],
    "DATE": [],
    "MONEY": [],
    "GPE": []
  },
  "sentiment": "positive"
}
```

## Deploy on Render

### Render setup

- Create a new **Web Service** from your Git repo
- Set Root Directory to `generated_backend` (or deploy that folder as repo root)
- Runtime: **Python 3.10+**

### Environment variables

- `API_KEY`: your secret API key

### Build command

```bash
pip install -r requirements.txt && python -m spacy download en_core_web_sm
```

### Start command

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## AI tools used

This project can be built with assistance from tools like ChatGPT / Windsurf for scaffolding and iterative development.

## Known limitations

- OCR quality depends heavily on image quality and Tesseract installation.
- PDF extraction is text-based; scanned PDFs require OCR (not implemented as a PDF OCR pipeline).
- Summary is a simple extractive summary (first few sentences) for speed and stability.
