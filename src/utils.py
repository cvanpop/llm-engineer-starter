from openai import OpenAI
from typing import List
import logging

# Initialize the OpenAI client
client = OpenAI()


def get_embedding(text: str) -> List[float]:
    """
    Get the embedding for the given text using OpenAI's API.

    This function performs the following steps:
    1. Sends a request to OpenAI's API to create an embedding for the input text
    2. Extracts the embedding from the API response
    3. Returns the embedding as a list of floats

    Args:
        text (str): The input text to embed.

    Returns:
        List[float]: The embedding vector, a list of 1536 floating-point numbers
                     for the "text-embedding-ada-002" model.

    Raises:
        RuntimeError: If there's an error getting the embedding from the API.
    """
    try:
        # Make a request to OpenAI's API to create an embedding
        response = client.embeddings.create(
            input=[text],  # The API expects a list of strings, even for a single input
            model='text-embedding-ada-002'
        )

        # Extract the embedding from the response
        # The API returns a list of embeddings (one per input text)
        # Take the first (and only) embedding and return its vector
        return response.data[0].embedding

    except Exception as e:
        # If any error occurs during the API call or processing the response,
        # log the error and raise a RuntimeError
        logging.error(f"Error getting embedding: {str(e)}")
        raise RuntimeError(f"Error getting embedding: {str(e)}")
