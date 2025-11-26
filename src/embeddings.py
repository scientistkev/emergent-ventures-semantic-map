"""
Embedding generation using OpenAI API.
"""

import os
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client with API key from environment.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in your .env file or environment variables."
        )
    return OpenAI(api_key=api_key)


def generate_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Generate embedding for a single text using OpenAI API.
    
    Args:
        text: Text to embed
        client: OpenAI client instance
        model: Model to use for embeddings
        
    Returns:
        Numpy array of embedding vector
    """
    # Replace newlines with spaces
    text = text.replace("\n", " ").strip()
    
    response = client.embeddings.create(
        model=model,
        input=text
    )
    
    return np.array(response.data[0].embedding)


def generate_embeddings_batch(
    texts: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-large",
    batch_size: int = 100
) -> np.ndarray:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to embed
        client: OpenAI client instance
        model: Model to use for embeddings
        batch_size: Number of texts to process per batch
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Clean texts
        batch_cleaned = [text.replace("\n", " ").strip() for text in batch]
        
        response = client.embeddings.create(
            model=model,
            input=batch_cleaned
        )
        
        batch_embeddings = [np.array(item.embedding) for item in response.data]
        embeddings.extend(batch_embeddings)
        
        print(f"Processed batch {i // batch_size + 1} / {(len(texts) + batch_size - 1) // batch_size}")
    
    return np.array(embeddings)


def generate_embeddings_for_dataset(
    data: List[Dict[str, Any]],
    output_path: Path,
    text_field: str = "embedding_text",
    model: str = "text-embedding-3-large"
) -> np.ndarray:
    """
    Generate embeddings for all entries in dataset.
    
    Args:
        data: List of entry dictionaries
        output_path: Path to save embeddings numpy array
        text_field: Field name containing text to embed
        model: OpenAI model to use
        
    Returns:
        Numpy array of embeddings
    """
    client = get_openai_client()
    
    # Extract texts
    texts = [entry.get(text_field, "") for entry in data]
    
    print(f"Generating embeddings for {len(texts)} entries using {model}...")
    
    # Generate embeddings
    embeddings = generate_embeddings_batch(texts, client, model=model)
    
    # Validate embeddings
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Check for NaN or invalid values
    if np.isnan(embeddings).any():
        raise ValueError("Found NaN values in embeddings!")
    
    if np.isinf(embeddings).any():
        raise ValueError("Found infinite values in embeddings!")
    
    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to: {output_path}")
    
    return embeddings

