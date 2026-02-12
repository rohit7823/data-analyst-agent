"""
Persistent Semantic Memory using ChromaDB + Ollama Embeddings.
Stores conversation insights as embeddings for cross-session retrieval.
"""

import os
import hashlib
import logging
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom embedding function using Ollama's local embedding API."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
    
    def name(self) -> str:
        return f"ollama_{self.model}"
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using Ollama's API."""
        embeddings = []
        for text in texts:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                    timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embeddings"][0])
            except Exception as e:
                logger.error(f"Embedding failed for text: {e}")
                embeddings.append([0.0] * 768)
        return embeddings
    
    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(input)
    

class MemoryStore:
    """
    Persistent semantic memory backed by ChromaDB.
    
    Stores conversation insights and retrieves relevant context
    via similarity search before each LLM call.
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.memory_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Parse Ollama base URL (strip /v1 suffix if present)
        ollama_url = settings.ollama_base_url.rstrip("/")
        if ollama_url.endswith("/v1"):
            ollama_url = ollama_url[:-3]
        
        self.embedding_fn = OllamaEmbeddingFunction(
            model=settings.embedding_model,
            base_url=ollama_url
        )
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Single shared collection for all sessions
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"MemoryStore initialized: {self.collection.count()} memories in {self.persist_dir}")
    
    def add(self, text: str, metadata: Optional[dict] = None, session_id: Optional[str] = None) -> str:
        """
        Store a memory with optional metadata.
        Returns the memory ID.
        """
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id or "global",
        }
        if metadata:
            # Flatten metadata — ChromaDB only accepts str/int/float/bool
            for k, v in metadata.items():
                meta[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
        
        try:
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta]
            )
            logger.info(f"Memory added [{doc_id}]: {text[:80]}...")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
        
        return doc_id
    
    def search(self, query: str, top_k: Optional[int] = None, session_id: Optional[str] = None) -> list[dict]:
        """
        Search for relevant memories using semantic similarity.
        Returns list of {text, score, metadata} dicts.
        """
        if self.collection.count() == 0:
            return []
        
        k = min(top_k or settings.memory_top_k, self.collection.count())
        
        try:
            where_filter = None
            if session_id:
                where_filter = {"session_id": session_id}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter if where_filter else None
            )
            
            memories = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    memories.append({
                        "text": doc,
                        "score": round(1 - results["distances"][0][i], 3) if results["distances"] else 0,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    def save_conversation_turn(
        self,
        user_message: str,
        agent_response: str,
        session_id: Optional[str] = None,
        file_name: Optional[str] = None
    ):
        """
        Save a Q&A turn as a memory. Combines question + answer into
        a single document for better retrieval.
        """
        # Skip very short or error responses
        if len(agent_response) < 20 or agent_response.startswith("Error"):
            return
        
        # Create a concise memory document
        memory_text = f"Q: {user_message}\nA: {agent_response[:500]}"
        
        metadata = {"type": "conversation"}
        if file_name:
            metadata["file_name"] = file_name
        
        self.add(memory_text, metadata=metadata, session_id=session_id)
    
    def get_context_for_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Retrieve and format relevant memories as context for the system prompt.
        """
        memories = self.search(query, session_id=session_id)
        
        if not memories:
            return ""
        
        lines = ["## Relevant Past Context"]
        for i, mem in enumerate(memories, 1):
            score = mem["score"]
            if score < 0.3:  # Skip low-relevance memories
                continue
            lines.append(f"\n### Memory {i} (relevance: {score})")
            lines.append(mem["text"])
        
        if len(lines) == 1:  # Only header, no relevant memories
            return ""
        
        return "\n".join(lines)
    
    def count(self) -> int:
        """Return total number of stored memories."""
        return self.collection.count()
