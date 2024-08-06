import ast
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Union
from difflib import SequenceMatcher
import logging
from openai import OpenAI
from src.utils import get_embedding
from src.indexer import normalize_vector

# Initialize OpenAI client
client = OpenAI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_blocks_with_substring(
    blocks: List[Dict], substring: str, similarity_threshold: float = 0.75
) -> List[Dict]:
    """
    Find blocks that contain a given substring or text with high similarity.

    This function performs two types of searches:
    1. Exact match: It first looks for blocks that contain the exact substring.
    2. Similarity match: If no exact matches are found, it searches for blocks
       with text similar to the substring, using a similarity threshold.

    The similarity is calculated using the SequenceMatcher ratio, which compares
    sequences of characters and returns a value between 0 and 1, where 1 means
    perfect similarity.

    Args:
        blocks (List[Dict]): A list of block dictionaries. Each dictionary should
                             contain at least a 'text' key with the block's content.
        substring (str): The substring to search for within the blocks.
        similarity_threshold (float, optional): The minimum similarity ratio required
                                                for a block to be considered a match
                                                when no exact matches are found.
                                                Defaults to 0.75.

    Returns:
        List[Dict]: A list of block dictionaries that either contain the exact
                    substring or have text similar to the substring above the
                    specified threshold. Returns an empty list if no matches are found.

    Note:
        - The function converts both the substring and block text to lowercase
          before comparison to ensure case-insensitive matching.
        - For similarity matching, it compares substrings of the block text
          to handle cases where the similar text is part of a larger block.
    """
    substring = substring.lower()
    result = []

    def similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    logging.info(f"Searching for substring: '{substring}'")
    logging.info(f"Number of blocks to search: {len(blocks)}")

    for block in blocks:
        block_text = block["text"].lower()
        # Exact string match
        if substring in block_text:
            block["sequence_matcher_score"] = 1.0  # Perfect match
            result.append(block)
            continue
        # Similarity match via SequenceMatcher
        if not result:
            max_similarity = 0
            for i in range(len(block_text) - min(len(substring), 10) + 1):
                chunk = block_text[i : i + min(len(substring), 20)]
                similarity = similar(chunk, substring)
                if similarity > max_similarity:
                    max_similarity = similarity
            if max_similarity > similarity_threshold:
                block["sequence_matcher_score"] = max_similarity
                result.append(block)

    logging.info(f"Total blocks found: {len(result)}")
    return result


def filter_chunks_by_similarity_cosine(
    chunks_with_metadata: List[Tuple[str, List[Dict], str]],
    S: np.ndarray,
    I: np.ndarray,
    similarity_threshold: float = 0.5,
) -> List[Tuple[Tuple[str, List[Dict], str], float]]:
    """
    Filter chunks based on cosine similarity.

    Args:
        chunks_with_metadata: List of chunks with metadata.
        S: Similarity scores.
        I: Indices of similar chunks.
        similarity_threshold: Threshold for filtering chunks.

    Returns:
        List of filtered chunks with their similarity scores.
    """
    # Initialize an empty list to store filtered chunks
    filtered_chunks = []

    # Iterate through the indices and similarity scores
    for i, (chunk_index, similarity_score) in enumerate(zip(I[0], S[0])):
        # Ensure the cosine similarity is between 0 and 1
        cosine_sim = min(max(similarity_score, 0), 1)

        # If the cosine similarity is above the threshold, add the chunk to the filtered list
        if cosine_sim >= similarity_threshold:
            # Append a tuple containing the chunk (with its metadata) and its similarity score
            filtered_chunks.append((chunks_with_metadata[chunk_index], cosine_sim))

    # Return the list of filtered chunks with their similarity scores
    return filtered_chunks


def extract_json_from_code_block(text: str) -> Union[Dict, None]:
    """
    Extract JSON from a code block in the text.

    Args:
        text (str): The text containing a potential JSON code block.

    Returns:
        Dict | None: Extracted JSON object or None if extraction fails.
    """
    if text.strip().startswith("```"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logging.error("The extracted string is not valid JSON.")
    return None


def remove_duplicates(list_of_dicts: List[Dict]) -> List[Dict]:
    """
    Remove duplicate dictionaries from a list.

    Args:
        list_of_dicts (List[Dict]): A list of dictionaries.

    Returns:
        List[Dict]: A list of unique dictionaries.
    """

    def make_hashable(obj):
        if isinstance(obj, (tuple, list)):
            return tuple(make_hashable(e) for e in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, set):
            return tuple(sorted(make_hashable(e) for e in obj))
        return obj

    seen = set()
    unique_dicts = []
    for d in list_of_dicts:
        hashable = make_hashable(d)
        if hashable not in seen:
            seen.add(hashable)
            unique_dicts.append(d)

    return unique_dicts

# If sorting by location is preferred.
# def sort_blocks(blocks: List[Dict]) -> List[Dict]:
#     """
#     Sort blocks based on their page number and position.

#     Args:
#         blocks (List[Dict]): A list of block dictionaries.

#     Returns:
#         List[Dict]: A sorted list of block dictionaries.
#     """

#     def get_sort_key(block):
#         page = int(block.get("page", 0))
#         if "vertices" in block and block["vertices"]:
#             y_coord = block["vertices"][0][1]
#             x_coord = block["vertices"][0][0]
#         else:
#             y_coord = x_coord = 0
#         return (page, y_coord, x_coord)

#     return sorted(blocks, key=get_sort_key)


def answer_question(
    index: faiss.IndexFlatIP,
    chunks_with_metadata: List[Tuple[str, List[Dict], str]],
    question: str,
    k: int,
    similarity_threshold: float = 0.5,
) -> Dict:
    """
    Answer a question based on the provided index and chunks.

    This function performs the following steps:
    1. Generate and normalize the embedding for the input question.
    2. Perform a similarity search using FAISS to find the k most similar chunks.
    3. Filter and sort the retrieved chunks based on similarity scores.
    4. Extract relevant chunks and their texts.
    5. Generate an answer using GPT-4o, providing the relevant chunk texts as context.
    6. Parse the generated answer and extract substrings used in the answer.
    7. Find relevant blocks for each substring within all retrieved chunks.
    8. Remove duplicate blocks from the set of relevant blocks.
    9. Sort the blocks by equence_matcher_score
    10. Return a dictionary containing the answer, substrings, and scored relevant blocks.

    Args:
        index (faiss.IndexFlatIP): The FAISS index for similarity search.
        chunks_with_metadata (List[Tuple[str, List[Dict], str]]): A list of chunks with their metadata.
        question (str): The question to answer.
        k (int): The number of chunks to retrieve.
        similarity_threshold (float): Threshold for filtering chunks.

    Returns:
        Dict: A dictionary containing the answer, substings, and relevant blocks.
    """
    # Generate and normalize the embedding for the question
    question_embedding = normalize_vector(get_embedding(question))

    # Check the total number of vectors in the index
    total_vectors = index.ntotal
    
    # Use the smaller of k and total_vectors
    # Faiss will return k results even if len of index is less than k.
    k_adjusted = min(k, total_vectors)
    
    # Perform similarity search using FAISS
    D, I = index.search(np.array([question_embedding], dtype=np.float32), k_adjusted)

    # Filter and sort chunks based on similarity scores
    filtered_chunks = filter_chunks_by_similarity_cosine(
        chunks_with_metadata, D, I, similarity_threshold
    )
    filtered_chunks.sort(key=lambda x: x[1], reverse=True)
    # Extract relevant chunks and their texts
    relevant_chunks = [chunk_tuple for chunk_tuple, _ in filtered_chunks]
    chunk_texts = [text for text, _, _ in relevant_chunks]

    # Generate an answer using GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. You help medical providers answer questions about their patients. Your answer is really important for the provider and patient. Let's think about this step by step. Answer the question based on the provided context of text. Note that the context is a list of texts. Please keep track of which items in the context lists that you use in your answer. Identify the substrings used to answer the question. Return your answer as a dictionary of the following format:
                {'answer': str. the answer to the question,
                "substrings": list of strings. a list of substrings that you used to formulate your answer. Be sure to reproduce the text exactly. Be concise.
                }
                If the context does not answer the question, return the following:
                {'answer': 'There is no information on this topic'. # you can rephrase this as needed. The point is to communicate that you cannot answer the question based on the documents provided.,
                'substrings': [] #empty list
                }

                It is better to provide as much of an answer as possible rather than no answer. 
                
                """,
            },
            {
                "role": "user",
                "content": f"Context: {chunk_texts}\n\nQuestion: {question}",
            },
        ],
    )

    # Extract and log the answer text
    answer_text = response.choices[0].message.content
    logging.info(f"answer_text: {answer_text}")

    # Parse the answer text into a dictionary
    answer_dict = extract_json_from_code_block(answer_text)
    if answer_dict is None:
        try:
            answer_dict = ast.literal_eval(answer_text)
        except (SyntaxError, ValueError):
            logging.error("Failed to parse answer_text. Using default values.")
            answer_dict = {"answer": "", "substrings": []}

    if not isinstance(answer_dict, dict):
        logging.error("Parsed result is not a dictionary. Using default values.")
        answer_dict = {"answer": "", "substrings": []}

    # Extract answer and substrings from the parsed dictionary
    answer = answer_dict.get("answer", "")
    substrings = answer_dict.get("substrings", [])
    substrings = list(set(substrings))  # Remove duplicates

    logging.info(f"answer: {answer}")
    logging.info(f"substrings: {substrings}")
    
    # Find relevant blocks for each substring
    all_relevant_blocks = []
    for substring in substrings:
        for chunk, blocks, _ in relevant_chunks:
            relevant_blocks = find_blocks_with_substring(blocks, substring)
            all_relevant_blocks.extend(relevant_blocks)

    # Remove duplicate blocks
    unique_relevant_blocks = remove_duplicates(all_relevant_blocks)
    logging.info(f"Number of unique relevant blocks: {len(unique_relevant_blocks)}")

    # Sort blocks by sequence_matcher_score
    sorted_blocks = sorted(unique_relevant_blocks, key=lambda x: x.get('sequence_matcher_score', 0), reverse=True)
    
    logging.info(f"Number of sorted blocks: {len(sorted_blocks)}")

    # Prepare and return the final result
    return {
        "answer": answer,
        "substrings": substrings,
        "blocks": sorted_blocks,
    }


def answer_question_short_doc(
    full_text: str, question: str, block_mapping: List[Dict]
) -> Dict:
    """
    Answer a question for a short document without using chunking or indexing.
    Args:
        full_text (str): The full text of the document.
        question (str): The question to answer.
        block_mapping (List[Dict]): A list of block dictionaries containing metadata.
    Returns:
        Dict: A dictionary containing the answer, substings, and relevant blocks.
    """
    # Call GPT-4o API to generate an answer
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. You help medical providers answer questions about their patients. Your answer is really important for the provider and patient. Let's think about this step by step. Answer the question based on the provided context of text. Note that the context is a list of texts. Please keep track of which items in the context lists that you use in your answer. Identify the substrings used to answer the question. Return your answer as a dictionary of the following format:
                {'answer': str. the answer to the question,
                "substrings": list of strings. a list of substrings that you used to formulate your answer. Be sure to reproduce the text exactly. Be concise.
                }
                If the context does not answer the question, return the following:
                {'answer': 'There is no information on this topic'. # you can rephrase this as needed. The point is to communicate that you cannot answer the question based on the documents provided.,
                'substrings': [] #empty list
                }

                It is better to provide as much of an answer as possible rather than no answer. 
                """
            },
            {
                "role": "user",
                "content": f"Context: {full_text}\n\nQuestion: {question}",
            },
        ],
    )

    # Extract the answer text from the API response
    answer_text = response.choices[0].message.content
    logging.info(f"answer_text: {answer_text}")

    # Try to parse the answer text as JSON
    answer_dict = extract_json_from_code_block(answer_text)
    if answer_dict is None:
        try:
            # If JSON parsing fails, try to evaluate as a Python literal
            answer_dict = ast.literal_eval(answer_text)
        except (SyntaxError, ValueError):
            # If both parsing methods fail, use default values
            logging.error("Failed to parse answer_text. Using default values.")
            answer_dict = {"answer": "", "substrings": []}

    # Ensure the parsed result is a dictionary
    if not isinstance(answer_dict, dict):
        logging.error("Parsed result is not a dictionary. Using default values.")
        answer_dict = {"answer": "", "substrings": []}

    # Extract answer and substrings from the parsed dictionary
    answer = answer_dict.get("answer", "")
    substrings = answer_dict.get("substrings", [])

    # Remove duplicate substrings
    substrings = list(set(substrings))
    logging.info(f"answer: {answer}")
    logging.info(f"substrings: {substrings}")

     # Find relevant blocks for each substring
    all_relevant_blocks = []
    for substring in substrings:
        logging.info(f"\n  Searching for substring: '{substring}'")
        relevant_blocks = find_blocks_with_substring(block_mapping, substring)
        if relevant_blocks:
            logging.info(f"  Found {len(relevant_blocks)} relevant blocks")
            all_relevant_blocks.extend(relevant_blocks)
        else:
            logging.info("No specific blocks found, searching in entire document")
            if substring.lower() in full_text.lower():
                logging.info("Substring found in document, including all blocks")
                for block in block_mapping:
                    block["sequence_matcher_score"] = 1.0  # Assign a score for all blocks
                all_relevant_blocks.extend(block_mapping)
            else:
                logging.info("Substring not found in document")

    # Remove duplicate blocks
    unique_relevant_blocks = remove_duplicates(all_relevant_blocks)
    logging.info(f"Number of unique relevant blocks: {len(unique_relevant_blocks)}")

    # Sort blocks by sequence_matcher_score
    sorted_blocks = sorted(unique_relevant_blocks, key=lambda x: x.get('sequence_matcher_score', 0), reverse=True)
    
    logging.info(f"Number of sorted blocks: {len(sorted_blocks)}")

    # Return the final result
    return {
        "answer": answer,
        "substrings": substrings,
        "blocks": sorted_blocks,
    }