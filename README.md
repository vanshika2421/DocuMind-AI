# DocuMind AI — RAG Document Intelligence MVP

A working starter application for document-grounded question answering using FastAPI, OpenAI, Pinecone and a lightweight web UI.

## What works
- PDF and DOCX upload
- Text extraction (PDF page-level; DOCX document-level locator)
- Overlapping character-based chunking
- OpenAI `text-embedding-3-small` embeddings
- Pinecone vector index creation and upsert
- Semantic retrieval and GPT-4.1 grounded answers
- Retrieved source previews with document and page/section labels
- Delete-by-document-ID API

## Requirements
- Python 3.10+
- OpenAI API key with API billing/credits enabled
- Pinecone API key

## Quick start (Windows PowerShell)

```powershell

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Create your local environment file
Copy-Item .env.example .env
notepad .env
```

Add your real keys to `.env`. Do not share `.env`, commit it, or paste API keys into chat.

```powershell
# 4. Start the application
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## API
- `GET /api/health` — API/key configuration status
- `POST /api/documents` — multipart upload field `file`
- `POST /api/chat` — JSON body `{"question":"..."}`
- `DELETE /api/documents/{document_id}` — delete vectors belonging to a document
- `GET /docs` — interactive Swagger API docs

## Important implementation notes
- The first upload creates a Pinecone serverless index named `documind-ai` in AWS `us-east-1` if it does not exist. Change `PINECONE_INDEX` if you need a different index. Index dimension must match the embedding model (1536 for `text-embedding-3-small` here).
- Pinecone index creation can take a short time on the first run.
- Chunk size and overlap are character counts, not token counts. Tune them after evaluating retrieval quality.
- PDF extraction works for text-based PDFs. Scanned/image-only PDFs require OCR, not included in this MVP.
- DOCX citations identify the document rather than exact Word page numbers because DOCX pagination depends on rendering.
- The MVP uses one shared Pinecone namespace and does not yet implement authentication or per-user isolation. Do not deploy it with sensitive documents as-is.
- Retrieved passages are shown as sources; similarity scores are not calibrated confidence probabilities.
- The app sends extracted document text to OpenAI for embeddings and relevant retrieved text to the chat model. Do not upload confidential data unless your data-handling requirements permit it.
- This is a starter MVP, not a production security-reviewed service.

## Next engineering steps
1. Add persistent document registry and upload status in PostgreSQL.
2. Add auth and tenant-scoped vector namespaces/filters.
3. Add robust duplicate handling and safe file lifecycle.
4. Add retrieval evaluation dataset and Recall@K / MRR.
5. Add automated tests, logging, rate limiting and Docker deployment.
6. Port the pipeline to C#/.NET if retaining the resume's original stack is required.
