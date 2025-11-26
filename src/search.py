"""
Semantic search using cosine similarity over embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import get_openai_client, generate_embedding


class SemanticSearch:
    """
    Semantic search engine for Emergent Ventures dataset.
    """
    
    def __init__(self, data: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Initialize semantic search engine.
        
        Args:
            data: List of entry dictionaries
            embeddings: Numpy array of embeddings (n_entries, embedding_dim)
        """
        self.data = data
        self.embeddings = embeddings
        self.client = None  # Lazy initialization
        
        if len(data) != embeddings.shape[0]:
            raise ValueError(
                f"Data length ({len(data)}) doesn't match embeddings shape ({embeddings.shape[0]})"
            )
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self.client is None:
            self.client = get_openai_client()
        return self.client
    
    def embed_query(self, query: str, model: str = "text-embedding-3-large") -> np.ndarray:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query string
            model: OpenAI model to use
            
        Returns:
            Query embedding vector
        """
        client = self._get_client()
        return generate_embedding(query, client, model=model)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        model: str = "text-embedding-3-large"
    ) -> List[Dict[str, Any]]:
        """
        Search for most similar entries to a query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            model: OpenAI model to use for query embedding
            
        Returns:
            List of result dictionaries with similarity scores
        """
        # Embed query
        query_embedding = self.embed_query(query, model=model)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = self.data[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            result['rank'] = len(results) + 1
            results.append(result)
        
        return results
    
    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with similarity scores
        """
        query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = self.data[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            result['rank'] = len(results) + 1
            results.append(result)
        
        return results
    
    def find_similar_entries(
        self,
        entry_index: int,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find entries similar to a given entry.
        
        Args:
            entry_index: Index of entry in dataset
            top_k: Number of results to return
            exclude_self: Whether to exclude the query entry from results
            
        Returns:
            List of similar entry dictionaries
        """
        query_embedding = self.embeddings[entry_index:entry_index+1]
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1]
        
        if exclude_self:
            top_indices = top_indices[top_indices != entry_index]
        
        top_indices = top_indices[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            result = self.data[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            result['rank'] = len(results) + 1
            results.append(result)
        
        return results


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Formatted string
    """
    lines = []
    for result in results:
        lines.append(f"\nRank {result['rank']} (Similarity: {result['similarity_score']:.4f})")
        lines.append(f"  Name: {result.get('name', 'N/A')}")
        lines.append(f"  Project: {result.get('project_name', 'N/A')}")
        lines.append(f"  Description: {result.get('project_description', 'N/A')}")
        lines.append(f"  Domains: {', '.join(result.get('domains', []))}")
        lines.append(f"  Category: {result.get('category', 'N/A')}")
    
    return "\n".join(lines)

