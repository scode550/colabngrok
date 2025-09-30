import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles embedding creation, storage, and retrieval using FAISS."""

    def __init__(self, data_dir="data", model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "document.index")
        self.metadata_path = os.path.join(data_dir, "document.metadata")

        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        """Loads the FAISS index and metadata from disk if they exist."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors.")
            else:
                # Initialize a new index if one doesn't exist
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Initialized a new FAISS index.")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}. Re-initializing.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []

    def _save(self):
        """Saves the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved FAISS index and metadata to disk.")

    def add_documents(self, chunks: list[str], filename: str):
        """
        Creates embeddings for document chunks and adds them to the FAISS index.

        Args:
            chunks: A list of text chunks.
            filename: The name of the source document.
        """
        if not chunks:
            return
            
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        embeddings_np = np.array(embeddings, dtype='float32')
        
        self.index.add(embeddings_np)
        
        # Store metadata for each chunk
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "source": f"{filename}_chunk_{len(self.metadata)}",
                "content": chunk
            })
            
        self._save()
        logger.info(f"Added {len(chunks)} new vectors to the index. Total vectors: {self.index.ntotal}")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Performs a similarity search for a given query.

        Args:
            query: The user's query text.
            k: The number of top results to return.

        Returns:
            A list of dictionaries, where each dictionary contains the 'source' and 'content'.
        """
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty index.")
            return []
            
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding_np = np.array(query_embedding, dtype='float32')
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        results = [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        return results
