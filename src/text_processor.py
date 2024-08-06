import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_text(text: str) -> str:
    """
    Clean the given text by removing extra whitespace.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text with extra whitespace removed.
    """
    return " ".join(text.split())


def chunk_text_with_metadata(
    text: str, block_mapping: List[Dict]
) -> List[Tuple[str, List[Dict], str]]:
    """
    Split the given text into chunks and associate each chunk with its corresponding metadata.

    This function performs the following steps:
    1. Initialize a RecursiveCharacterTextSplitter with specific parameters.
    2. Split the input text into chunks.
    3. For each unique chunk:
       a. Generate a unique identifier.
       b. Find its position in the original text.
       c. Identify relevant blocks from the block_mapping that overlap with the chunk.
       d. Create a tuple of (chunk, relevant_blocks, chunk_id).
    4. Return a list of these tuples.

    Args:
        text (str): The input text to split into chunks.
        block_mapping (List[Dict]): A list of dictionaries containing metadata for each block of text.

    Returns:
        List[Tuple[str, List[Dict], str]]: A list of tuples, where each tuple contains:
            - A chunk of text (str)
            - Its corresponding blocks (List[Dict])
            - A unique identifier (str)
    """
    # Initialize the text splitter with specific parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #char length
        chunk_overlap=100, #char length
        length_function=len,
    )

    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    logging.info(f"Number of chunks created: {len(chunks)}")

    unique_chunks_with_metadata = []
    unique_chunk_texts = set()

    for chunk in chunks:
        if chunk not in unique_chunk_texts:
            # Generate a unique identifier for each chunk for use at scale
            chunk_id = str(uuid.uuid4())

            # Find the position of this chunk in the original text
            chunk_start = text.index(chunk)
            chunk_end = chunk_start + len(chunk)

            # Identify relevant blocks that overlap with this chunk
            relevant_blocks = [
                block
                for block in block_mapping
                if block["char_start"] <= chunk_end and block["char_end"] >= chunk_start
            ]

            # Create a tuple of (chunk, relevant_blocks, chunk_id) and add it to the list
            unique_chunks_with_metadata.append((chunk, relevant_blocks, chunk_id))
            unique_chunk_texts.add(chunk)

    logging.info(f"Number of unique chunks: {len(unique_chunks_with_metadata)}")
    return unique_chunks_with_metadata
