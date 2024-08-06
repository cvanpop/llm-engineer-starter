import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict
from .schemas import ParsedDocument
from .pdf import DocumentParser


async def process_pdf(input_path: Path) -> ParsedDocument:
    """
    Process a PDF file and create a ParsedDocument object.

    This function performs the following steps:
    1. Creates a DocumentParser instance
    2. Processes the PDF file
    3. Returns the parsed document

    Args:
        input_path (Path): The path to the PDF file.

    Returns:
        ParsedDocument: The parsed document object.

    Raises:
        RuntimeError: If there's an error processing the PDF.
    """
    # Create a DocumentParser instance
    document_parser = DocumentParser()
    try:
        # Process the PDF file
        parsed_document = await document_parser.process_document(input_path)

        # Log information about the parsed document
        logging.info(
            f"ParsedDocument created with {len(parsed_document.pages)} page(s)"
        )
        # for i, page in enumerate(parsed_document.pages): # debuging.
        #     logging.info(f"Page {i+1} has {len(page.blocks)} blocks")

        return parsed_document
    except Exception as e:
        # If any error occurs during processing, raise a RuntimeError
        raise RuntimeError(f"Error processing PDF: {str(e)}")


def parsed_document_to_json(parsed_document: ParsedDocument) -> str:
    """
    Convert a ParsedDocument object to a JSON string.

    This function creates a dictionary representation of the ParsedDocument
    and then converts it to a JSON string.

    Args:
        parsed_document (ParsedDocument): The parsed document object.

    Returns:
        str: A JSON string representation of the parsed document.
    """
    # Create a dictionary representation of the ParsedDocument
    document_dict = {
        "total_pages": len(parsed_document.pages),
        "pages": [
            {
                "page_number": i + 1,
                "blocks": [
                    {"text": block.text, "vertices": block.vertices}
                    for block in page.blocks
                ],
            }
            for i, page in enumerate(parsed_document.pages)
        ],
    }

    # Convert the dictionary to a JSON string with indentation
    return json.dumps(document_dict, indent=2)


def create_full_text_and_mapping(
    parsed_document: ParsedDocument,
) -> Tuple[str, List[Dict]]:
    """
    Create full text and block mapping from a ParsedDocument.
    This function performs the following steps:
    1. Iterates through all pages and blocks in the parsed document
    2. Concatenates all block texts into a single string
    3. Creates a mapping of each block's position in the full text
    4. Adjusts page numbers to be 1-indexed

    Args:
        parsed_document (ParsedDocument): The parsed document object.

    Returns:
        Tuple[str, List[Dict]]: A tuple containing:
            - The full text of the document (str)
            - A list of dictionaries, each representing a block and its position in the full text
    """
    full_text = ""
    block_mapping = []

    for page_num, page in enumerate(parsed_document.pages):
        for block in page.blocks:
            # Record the start index of this block in the full text
            start_index = len(full_text)

            # Add the block's text to the full text, with a space
            full_text += block.text + " "

            # Record the end index of this block in the full text
            end_index = len(full_text)

            # Add this block's information to the mapping
            # Note: add 1 to page_num to make it 1-indexed to match pdf-indexing.
            # Same is done in process_and_save_json.
            block_mapping.append(
                {
                    "text": block.text,
                    "page": page_num + 1,  # Adjust to 1-indexed to match json.
                    "vertices": block.vertices,
                    "char_start": start_index,
                    "char_end": end_index,
                }
            )

    # Log information about the created full text and mapping
    logging.info(f"Full text length: {len(full_text)} characters")
    logging.info(f"Number of blocks: {len(block_mapping)}")

    # Return the full text and the block mapping
    return full_text, block_mapping
