import numpy as np
import faiss
from typing import List, Dict, Tuple
from src.utils import get_embedding

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector.

    This function performs the following steps:
    1. Calculates the L2 norm of the input vector
    2. If the norm is not zero, divides the vector by its norm
    3. Returns the normalized vector

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def create_index(
    chunks_with_metadata: List[Tuple[str, List[Dict], str]]
) -> faiss.IndexFlatIP:
    """
    Create a FAISS index from the given chunks and their metadata.

    This function performs the following steps:
    1. Generates embeddings for each chunk of text
    2. Normalizes each embedding vector
    3. Creates a FAISS IndexFlatIP (Inner Product) index
    4. Adds the normalized embeddings to the index

    Args:
        chunks_with_metadata: A list of tuples, where each tuple contains:
            - A chunk of text (str)
            - Its corresponding metadata (List[Dict])
            - A unique identifier (str)

    Returns:
        - The FAISS index (faiss.IndexFlatIP)

    """
    # Generate embeddings for all chunks and normalize them
    embeddings = [normalize_vector(get_embedding(chunk)) for chunk, _, _ in chunks_with_metadata]

    # Create a new FAISS index using Inner Product
    index = faiss.IndexFlatIP(len(embeddings[0]))

    # Add all embeddings to the FAISS index
    index.add(np.array(embeddings, dtype=np.float32))

    # Return both the FAISS index
    return index