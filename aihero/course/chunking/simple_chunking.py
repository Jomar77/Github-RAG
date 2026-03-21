from __future__ import annotations


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        chunk = seq[i : i + size]
        result.append({"start": i, "chunk": chunk})
        if i + size >= n:
            break

    return result


def chunk_documents(docs, content_key="content", size=2000, step=1000):
    chunks_out = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_key)
        chunks = sliding_window(doc_content, size, step)
        for chunk in chunks:
            chunk.update(doc_copy)
        chunks_out.extend(chunks)

    return chunks_out
