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

def delete_local_file_context(source_name: str, collection) -> bool:
    """
    Deletes all vectorized chunks belonging to a specific file name 
    from the ChromaDB collection using its metadata tags.
    """
    try:
        # ChromaDB allows filtering what to delete using a 'where' dictionary
        collection.delete(
            where={"source": source_name} # Matches the file name metadata tag
        )
        logger.info(f"Successfully deleted all vectors for file: {source_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file {source_name} from vector store: {e}")
        return False
        
def get_all_uploaded_files(collection) -> list:
    """
    Fetches all document metadata from the collection and extracts
    the unique list of original filenames currently stored.
    """
    try:
        # Retrieve metadata records from the collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        # Extract unique filenames from the metadata dictionaries
        unique_files = set()
        for meta in metadatas:
            if meta and "source" in meta:
                unique_files.add(meta["source"])
                
        return sorted(list(unique_files))
    except Exception as e:
        logger.error(f"Error fetching uploaded files list: {e}")
        return []

def delete_local_file_context(source_name: str, collection) -> bool:
    """
    Deletes all vectorized chunks belonging to a specific file name 
    from the ChromaDB collection using its metadata tags.
    """
    try:
        collection.delete(
            where={"source": source_name}
        )
        logger.info(f"Successfully deleted all vectors for file: {source_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file {source_name} from vector store: {e}")
        return False