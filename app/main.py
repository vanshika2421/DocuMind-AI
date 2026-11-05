
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings

from .rag import (
    ingest,
    answer_question,
    delete_document
)

from .memory import (
    initialize_memory,
    get_history,
    save_turn,
    clear_history,
    initialize_documents,
    save_document,
    list_documents,
    remove_document_record
)


# --------------------------------------------------
# APPLICATION INITIALIZATION
# --------------------------------------------------

app = FastAPI(
    title="DocuMind AI",
    version="0.1.0",
    description="RAG document intelligence MVP"
)


# Static frontend directory
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# --------------------------------------------------
# DATABASE STARTUP
# --------------------------------------------------

@app.on_event("startup")
def startup():
    initialize_memory()
    initialize_documents()


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=4000
    )

    session_id: str = Field(
        min_length=1,
        max_length=100
    )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
        "pinecone_configured": bool(settings.pinecone_api_key)
    }


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):

    filename = file.filename or "uploaded-document"

    # Validate file extension
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or DOCX file."
        )

    # Read uploaded file
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # Maximum upload size: 20 MB
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 20 MB."
        )

    try:
        # Ingest document into Pinecone
        result = ingest(filename, content)

        # Save document metadata in SQLite
        save_document(
            result["document_id"],
            result["filename"],
            result["chunks"]
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# LIST ALL DOCUMENTS
# --------------------------------------------------

@app.get("/api/documents")
def get_documents():
    return list_documents()


# --------------------------------------------------
# CHAT WITH DOCUMENTS
# --------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    try:
        # Retrieve previous conversation
        history = get_history(request.session_id)

        # Generate answer using RAG
        result = answer_question(
            request.question,
            history
        )

        # Save current conversation
        save_turn(
            request.session_id,
            request.question,
            result["answer"]
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# DELETE DOCUMENT
# --------------------------------------------------

@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str):

    try:
        # Delete vectors from Pinecone
        delete_document(document_id)

        # Delete document metadata from SQLite
        remove_document_record(document_id)

        return {
            "deleted": True,
            "document_id": document_id
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# CLEAR CHAT HISTORY
# --------------------------------------------------

@app.delete("/api/chat/{session_id}")
def delete_chat_history(session_id: str):

    try:
        clear_history(session_id)

        return {
            "cleared": True,
            "session_id": session_id
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
