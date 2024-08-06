import ast
import json
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_layout_aware_string(pages: List[Dict]) -> str:
    logging.info(f"Creating layout-aware string for {len(pages)} pages")
    full_text = ""
    for page_num, page in enumerate(pages, 1):
        logging.info(f"Processing page {page_num}")
        # sorted_blocks = sort_blocks(page['blocks'])
        
        # Initialize a 2D grid to represent the page
        grid_size = 100
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        for block_num, block in enumerate(page['blocks'], 1):

        # for block_num, block in enumerate(sorted_blocks, 1):
            text = block['text']
            x_start = int(block['vertices'][0][0] * grid_size)
            y_start = int(block['vertices'][0][1] * grid_size)
            
            logging.debug(f"Placing block {block_num} at position ({x_start}, {y_start})")
            
            # Place the text in the grid
            for i, char in enumerate(text):
                if char == '\n':
                    y_start += 1
                    x_start = int(block['vertices'][0][0] * grid_size)
                else:
                    if 0 <= y_start < grid_size and 0 <= x_start + i < grid_size:
                        grid[y_start][x_start + i] = char
        
        # Convert grid to string
        page_text = f"\n\n--- Page {page_num} ---\n\n"
        for row in grid:
            page_text += ''.join(row).rstrip() + '\n'
        
        full_text += page_text
        logging.info(f"Completed processing page {page_num}")
    
    logging.info("Layout-aware string creation completed")
    return full_text
