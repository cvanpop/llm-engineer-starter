import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
gcp_project_id = os.getenv('GCP_PROJECT_ID')
gcp_region = os.getenv('GCP_REGION')
google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
gcp_processor_id = os.getenv('GCP_PROCESSOR_ID')
openai_api_key = os.getenv('OPENAI_API_KEY')

import argparse
import asyncio
import json
import ast
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from src.pdf_processor import (
    process_pdf as process_pdf_internal,
    parsed_document_to_json,
    create_full_text_and_mapping,
)
from src.text_processor import chunk_text_with_metadata
from src.indexer import create_index
from src.question_answerer import answer_question, answer_question_short_doc
from src.layout_processor import create_layout_aware_string

# Configure logging to display informative messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
 
# Define a threshold for short documents (e.g., 1000 characters, estimated to be 140-200 words)
# Documents shorter than this will be processed differently
SHORT_DOC_THRESHOLD = 1000

async def process_pdf_and_save(pdf_path: str, create_text_file: bool = False) -> Tuple[str, str]:
    """
    Process a PDF file, save its contents as JSON, and optionally create a layout-aware text file.

    This function performs the following steps:
    1. Validates the input PDF file
    2. Processes the PDF to extract its content
    3. Converts the extracted content to JSON format
    4. Logs the JSON output
    5. Saves the JSON output to a file in the same location as the PDF
    6. Optionally creates a layout-aware text file in the same location as the PDF

    Args:
        pdf_path (str): The path to the PDF file.
        create_text_file (bool): Whether to create a layout-aware text file.

    Returns:
        Tuple[str, str]: A tuple containing the JSON output and the path to the text file (if created, else empty string).

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the specified file is not a valid PDF.
        RuntimeError: If there's an error processing the PDF.
    """
    input_path = Path(pdf_path)

    if not input_path.exists():
        raise FileNotFoundError(f"The file {input_path} does not exist")

    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"The file {input_path} is not a valid PDF file")

    try:
        parsed_document = await process_pdf_internal(input_path)
    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {str(e)}")

    json_output = parsed_document_to_json(parsed_document)

    logging.info("JSON output:")
    logging.info(json_output)

    json_output_path = input_path.with_suffix(".json")
    with open(json_output_path, "w") as f:
        f.write(json_output)
    logging.info(f"JSON output saved to: {json_output_path}")

    text_file_path = ""
    if create_text_file:
        text_file_path = input_path.with_suffix(".txt")
        # create_layout_aware_text_file(str(json_output_path), str(text_file_path))
        # logging.info(f"Layout-aware text file created: {text_file_path}")
        data = ast.literal_eval(json_output)
        # Create the layout-aware string
        layout_aware_text = create_layout_aware_string(data['pages'])
        # Save the result
        with open(text_file_path, 'w') as f:
            f.write(layout_aware_text)
            logging.info(f"Text output saved to: {text_file_path}")

    return json_output, str(text_file_path)

async def process_and_answer_question(
    pdf_path: str, question: str, k: int
) -> Dict[str, Any]:
    """
    Process a PDF file and answer a question based on its contents.

    This function performs the following steps:
    1. Validates the input PDF file
    2. Processes the PDF to extract its content
    3. Creates full text and block mapping from the parsed document
    4. Determines whether to use short or long document processing based on text length
    5. For long documents, chunks the text, creates an index, and retrieves relevant chunks
    6. For short documents, uses the full text as context
    7. Generates an answer to the question using the appropriate method

    Args:
        pdf_path (str): The path to the PDF file.
        question (str): The question to answer.
        k (int): The number of chunks to retrieve for context in long documents.

    Returns:
        Dict[str, Any]: The result containing the answer and relevant information.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the specified file is not a valid PDF.
        RuntimeError: If there's an error processing the PDF.
    """
    input_path = Path(pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"The file {input_path} does not exist")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"The file {input_path} is not a valid PDF file")

    try:
        parsed_document = await process_pdf_internal(input_path)
    except Exception as e:
        raise RuntimeError(f"Error processing PDF: {str(e)}")

    full_text, block_mapping = create_full_text_and_mapping(parsed_document)

    if len(full_text) >= SHORT_DOC_THRESHOLD:
        logging.info(
            f"Document is long (length: {len(full_text)}). Using indexing for context retrieval."
        )
        chunks_with_metadata = chunk_text_with_metadata(full_text, block_mapping)
        index = create_index(chunks_with_metadata)
        result = answer_question(index, chunks_with_metadata, question, k)
        
    else:
        logging.info(
            f"Document is short (length: {len(full_text)}). Using full text as context."
        )
        result = answer_question_short_doc(full_text, question, block_mapping)
        
    logging.info("Question answering result:")
    logging.info(json.dumps(result, indent=2))
    return result

def main():
    """
    Entrypoint to the submission

    This function:
    1. Sets up command-line argument parsing
    2. Validates the input arguments
    3. Calls the appropriate processing function based on whether a question is provided
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-case-pdf",
        metavar="path",
        type=str,
        help="Path to local test case with which to run your code",
    )
    parser.add_argument(
        "--question", type=str, help="Question to ask about the document (optional)"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--create-layout-text", action="store_true", help="Create layout-aware text file"
    )
    args = parser.parse_args()

    if not args.path_to_case_pdf:
        raise ValueError(
            "Please provide a path to the PDF file using --path-to-case-pdf"
        )

    if args.question:
        asyncio.run(
            process_and_answer_question(args.path_to_case_pdf, args.question, args.k)
        )
    else:
        asyncio.run(process_pdf_and_save(args.path_to_case_pdf, args.create_layout_text))

if __name__ == "__main__":
    main()
    
