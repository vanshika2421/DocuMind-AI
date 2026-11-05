
import hashlib
import io
import time

from google import genai
from google.genai import types, errors
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from docx import Document

from .config import settings


# -----------------------------
# CLIENT INITIALIZATION
# -----------------------------

def clients():
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")

    gemini_client = genai.Client(
        api_key=settings.gemini_api_key
    )

    pinecone_client = Pinecone(
        api_key=settings.pinecone_api_key
    )

    return gemini_client, pinecone_client


# -----------------------------
# GEMINI RETRY HELPER
# -----------------------------

def gemini_with_retry(operation, operation_name, max_attempts=4):
    for attempt in range(max_attempts):
        try:
            print(f"Starting Gemini operation: {operation_name}")
            result = operation()
            print(f"Gemini operation successful: {operation_name}")
            return result

        except errors.ServerError as e:
            print(
                f"Gemini error in {operation_name}: {e}"
            )

            if attempt == max_attempts - 1:
                raise

            wait_time = 2 ** (attempt + 1)

            print(
                f"Retrying {operation_name} in {wait_time} seconds..."
            )

            time.sleep(wait_time)


# -----------------------------
# DOCUMENT EXTRACTION
# -----------------------------

def extract_pages(filename: str, content: bytes) -> list[dict]:
    name = filename.lower()
    pages = []

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))

        for i, page in enumerate(reader.pages, start=1):
            pages.append({
                "page": i,
                "text": page.extract_text() or ""
            })

    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(content))

        text = "\n".join(
            p.text for p in doc.paragraphs if p.text.strip()
        )

        for table in doc.tables:
            for row in table.rows:
                text += "\n" + " | ".join(
                    cell.text for cell in row.cells
                )

        pages.append({
            "page": "Document",
            "text": text
        })

    else:
        raise ValueError("Only PDF and DOCX files are supported.")

    if not any(p["text"].strip() for p in pages):
        raise ValueError(
            "No extractable text found. Scanned PDFs need OCR."
        )

    return pages


# -----------------------------
# TEXT CHUNKING
# -----------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:

    text = " ".join(text.split())

    if not text:
        return []

    if overlap < 0 or overlap >= size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + size // 2:
                end = boundary

        chunks.append(text[start:end].strip())

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return [chunk for chunk in chunks if chunk]


# -----------------------------
# PINECONE INDEX
# -----------------------------

def ensure_index(pc):

    names = [item.name for item in pc.list_indexes()]

    if settings.pinecone_index not in names:

        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        for _ in range(30):

            desc = pc.describe_index(settings.pinecone_index)

            if desc.status["ready"]:
                break

            time.sleep(2)

        else:
            raise TimeoutError(
                "Pinecone index was not ready within 60 seconds."
            )

    return pc.Index(settings.pinecone_index)


# -----------------------------
# GEMINI EMBEDDINGS
# -----------------------------

def embed(
    gemini_client,
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:

    embeddings = []

    for text in texts:

        result = gemini_with_retry(
            lambda: gemini_client.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions
                )
            ),
            operation_name=f"Embedding ({task_type})"
        )

        embeddings.append(result.embeddings[0].values)

    return embeddings


# -----------------------------
# DOCUMENT INGESTION
# -----------------------------

def ingest(filename: str, content: bytes) -> dict:

    gemini_client, pc = clients()
    index = ensure_index(pc)

    pages = extract_pages(filename, content)

    document_id = hashlib.sha256(
        filename.encode() + content
    ).hexdigest()[:20]

    vectors = []
    chunk_number = 0

    for page in pages:

        chunks = chunk_text(
            page["text"],
            settings.chunk_size,
            settings.chunk_overlap
        )

        for offset in range(0, len(chunks), 64):

            batch = chunks[offset:offset + 64]

            embeddings = embed(
                gemini_client,
                batch,
                task_type="RETRIEVAL_DOCUMENT"
            )

            for text, vector in zip(batch, embeddings):

                vectors.append({
                    "id": f"{document_id}-{chunk_number}",
                    "values": vector,
                    "metadata": {
                        "document_id": document_id,
                        "document_name": filename,
                        "page": str(page["page"]),
                        "chunk": chunk_number,
                        "text": text
                    }
                })

                chunk_number += 1

    for offset in range(0, len(vectors), 100):

        index.upsert(
            vectors=vectors[offset:offset + 100]
        )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks": len(vectors)
    }

# -----------------------------
# SEMANTIC RETRIEVAL
# -----------------------------

def retrieve(question: str, top_k: int | None = None) -> list[dict]:

    gemini_client, pc = clients()
    index = ensure_index(pc)

    query_vector = embed(
        gemini_client,
        [question],
        task_type="RETRIEVAL_QUERY"
    )[0]

    result = index.query(
        vector=query_vector,
        top_k=top_k or settings.top_k,
        include_metadata=True
    )

    matches = []

    for match in result.matches:

        metadata = match.metadata or {}

        matches.append({
            "score": float(match.score or 0),
            "document": metadata.get("document_name", "Unknown document"),
            "page": metadata.get("page", "Unknown"),
            "text": metadata.get("text", ""),
            "document_id": metadata.get("document_id", "")
        })

    return matches


# -----------------------------
# GEMINI RAG ANSWER GENERATION
# -----------------------------


def answer_question(
    question: str,
    history: list[dict] | None = None
) -> dict:

    gemini_client, _ = clients()

    matches = retrieve(question)

    if not matches:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the indexed documents."
            ),
            "sources": []
        }

    context = "\n\n".join(
        f"[Source {i+1}: {m['document']}, "
        f"page/section {m['page']}]\n{m['text']}"
        for i, m in enumerate(matches)
    )

    conversation = ""

    if history:
        conversation = "\n".join(
            f"{item['role'].upper()}: {item['content']}"
            for item in history[-10:]
        )

    prompt = f"""
You are DocuMind AI, a document-grounded assistant.

CONVERSATION HISTORY:
{conversation}

CURRENT RETRIEVED DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
{question}

INSTRUCTIONS:

1. Use conversation history to understand follow-up questions
   and resolve references such as "it", "that project", or "those skills".

2. Use ONLY the current retrieved document context for factual
   claims about the uploaded documents.

3. Never treat previous AI answers as verified source material.

4. If the current context does not contain the answer,
   clearly say that the information could not be found.

5. Do not invent facts.

6. Cite supporting sources inline using [Source 1], [Source 2].

7. Give clear, structured and useful answers.
"""

    response = gemini_client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )

    return {
        "answer": response.text or "",
        "sources": [
            {
                "document": m["document"],
                "page": m["page"],
                "score": round(m["score"], 4),
                "preview": m["text"][:280]
            }
            for m in matches
        ]
    }



# -----------------------------
# DOCUMENT DELETION
# -----------------------------

def delete_document(document_id: str) -> None:

    _, pc = clients()
    index = ensure_index(pc)

    index.delete(
        filter={
            "document_id": {
                "$eq": document_id
            }
        }
    )
