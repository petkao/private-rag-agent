import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import settings

class LocalVaultIngestor:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_HOST,
            model=settings.PRIMARY_EMBEDDING
        )
        self.db_path = settings.DB_DIR
        self.vector_store = None

    def initialize_vault(self):
        """Secures a connection to the embedded vector database file block."""
        self.vector_store = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings,
            collection_name="user_private_vault"
        )

    def sync_vault_directory(self):
        """Finds any local text documents inside the vault folder and processes them securely."""
        self.initialize_vault()
        vault_path = settings.VAULT_DIR
        files = [f for f in os.listdir(vault_path) if f.endswith('.txt')]
        
        if not files:
            print("ℹ️  Local file vault empty. Drop .txt items inside data/vault/ to supplement memory maps.")
            return

        for file_name in files:
            file_p = os.path.join(vault_path, file_name)
            with open(file_p, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse document contents into chunk sizes optimized for small model context profiles
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            metadatas = [{"source": file_name} for _ in chunks]
            
            self.vector_store.add_texts(texts=chunks, metadatas=metadatas)
        print(f"✓ Locally vectorized and processed {len(files)} file records inside the embedded DB.")

    def query_local_context(self, query: str, k: int = 2) -> str:
        """Pulls closest contextual vectors without triggering outward network traffic requests."""
        if not self.vector_store:
            self.initialize_vault()
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n".join([r.page_content for r in results])
        except Exception:
            return ""