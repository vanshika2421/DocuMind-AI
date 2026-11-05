from app.rag import chunk_text

def test_chunk_text_returns_content():
    chunks = chunk_text("alpha beta gamma delta epsilon", size=12, overlap=3)
    assert chunks
    assert "".join(chunks).replace(" ", "")  # non-empty output

def test_empty_text():
    assert chunk_text("   ", size=10, overlap=2) == []
