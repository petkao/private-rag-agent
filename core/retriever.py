# core/retriever.py

def retrieve_local_context(query: str, col) -> str:
    """
    Queries the local serverless ChromaDB collection using native query_texts 
    auto-embedding pipelines and returns stitched string contexts.
    """
    try:
        # Pass raw string to query_texts; Chroma uses its pre-configured ef natively
        db_results = col.query(query_texts=[query], n_results=4)
        docs = db_results.get('documents', [[]])[0]
        return "\n---\n".join(docs) if docs else ""
    except Exception as e:
        # Optional: Log your exception here to terminal stdout if troubleshooting
        print(f"[Retriever Error]: {e}")
        return ""