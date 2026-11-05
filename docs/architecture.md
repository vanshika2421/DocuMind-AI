# Architecture

## Ingestion
Browser -> FastAPI upload endpoint -> PDF/DOCX extraction -> character chunking -> OpenAI embeddings -> Pinecone upsert.

Each vector stores `document_id`, `document_name`, `page`, `chunk`, and source text as metadata.

## Query
Browser -> FastAPI chat endpoint -> question embedding -> Pinecone top-k similarity search -> context assembly -> GPT-4.1 -> answer plus retrieved-source metadata.

## Current limitations
- No OCR for scanned PDFs.
- DOCX source locator is document-level, not a rendered page.
- No authentication, tenant isolation, or PostgreSQL registry yet.
- Chunking is character-based; production tuning should be token-aware and evaluation-driven.
