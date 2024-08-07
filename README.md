# Solution Overview: PDF Question Answering System

## Exercise Description


1. String (JSON) representation of PDF with coordinates
2. Question answering with one PDF citation of coordinates

[See the full description.](https://anterior.notion.site/Senior-AI-Engineer-Take-Home-Exercise-f43d63053da147ac926ca8501902a9c4)

## Overview of Solution

The overall pipeline is structured as follows:


        Parse PDF -> Recreate Full Text -> Text Chunking -> Vectorization -> Indexing -> Retrieval -> Answer Generation




## Example Usage

### Requirements


Please see requirements.txt for requirements.

### .env


Create a .env file containing the following:
```bash
GCP_PROJECT_ID=your_project_id_here

GCP_REGION=your_region_here

GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

GCP_PROCESSOR_ID=your_processor_id_here

OPENAI_API_KEY=your_openai_api_key_here
```

### Command Line

Note  that I chose to keep PDF processing and question answering as two separate workflows. This allows the use of PDF processing independent of the question answering system. With this, the question answering system processes the PDF. This can easily be reworked if, for example, it would be preferred to have question answering take the path to parsed PDF.

First, let's look at how to use the system from the command line:

```bash
# Process a PDF and save its content as .json and optionally .txt
python submission.py --path-to-case-pdf your_pdf_file.pdf --create-layout-text
```
The .json and .txt versions of the file will save at the same location as the PDF with the appropriate extension.

```bash
# Ask a question about the PDF
python submission.py --path-to-case-pdf path/to/your/document.pdf --question "What tests were performed?" --k 10
```

The `--k` parameter determines how many chunks of text to retrieve when answering a question. The default is 5, but this can be adjusted based on the complexity of the question and the length of the document.

### Notebook

Note  that I chose to keep PDF processing and question answering as two separate workflows. This allows the use of PDF processing independent of the question answering system. With this, the question answering system processes the PDF. This can easily be reworked if, for example, it would be preferred to have question answering take the path to parsed PDF.

Within a notebook, you can run the system as follows:



```python
# Load environment variables from .env file
import os
from dotenv import load_dotenv

load_dotenv()

# Access environment variables
gcp_project_id = os.getenv('GCP_PROJECT_ID')
gcp_region = os.getenv('GCP_REGION')
google_application_credentials = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
gcp_processor_id = os.getenv('GCP_PROCESSOR_ID')
openai_api_key = os.getenv('OPENAI_API_KEY')


# Import necessary functions from code
from submission import process_and_save_json, process_and_answer_question


# Example usage in Jupyter notebook
pdf_path = "data/inpatient_record.pdf"

# To process the PDF and save as JSON
json_output = await process_and_save_json(pdf_path)

# To process the PDF and save as JSON
# json_output = await process_pdf_and_save(pdf_path, create_text_file=True)

# To process the PDF and answer a question
question = "What tests were performed?"
answer_output = await process_and_answer_question(pdf_path, question, 10)

```

### Example JSON output

Here is the data strucutre
```bash
{
  "pages": [
    {
      "page_number": 1, #int
      "blocks": [] #dict
    },  
    ...
    {
      "page_number": n,
      "blocks": []
    }
  ]
}
```


### Example Block


```bash
{
  "text": "Sheet\n",
  "vertices": [
    [
      0.11846251040697098,
      0.15952381491661072
    ],
    [
      0.1915563941001892,
      0.15992063283920288
    ],
    [
      0.1915563941001892,
      0.17341269552707672
    ],
    [
      0.11846251040697098,
      0.17301587760448456
    ]
  ]
}
```

### Question Answering Example


 The output is a dictionary contains three keys:
```bash
{
  'answer': 'answer to the question',
  'substrings': [], # list of str. the substrings identified by LLM.
  'blocks':[] #list of blocks as above plus additional fields.
}


```

Note that for demo purposes, I have opted to to include `substrings` in the output. This will be explained below. I have included it to help illustrate the inner workings of the code base. In production, however, this could be removed.


Here is the answer to the question, `What tests were perfromed?` as shown above.


```bash
{'answer': 'The following tests were performed: bilateral mammogram, core biopsies, diagnostic mammogram/US, ultrasound, MRI-BREAST BIL W/WO CON, and 3D tomosynthesis mammogram.',
 'substrings': ['ULTRASOUND',
  'bilateral mammogram 7/14/23',
  '3D tomosynthesis mammogram',
  'core biopsies',
  'MRI-BREAST BIL W/WO CON',
  'R diagnostic mammogram/US'],
 'blocks': [{'text': 'PROCEDURE: The procedure site was marked and a timeout procedure was performed. The\npatient was prepped and draped in a sterile fashion. Local anesthesia was provided\n(lidocaine 1% without epinephrine). A 14-gauge coaxial spring-loaded core biopsy\nneedle was used. Under direct ultrasound guidance, the needle was guided into the\narea of concern. Multiple core biopsies were obtained. A T3 coil biopsy clip was\nplaced at the site. The needles were removed. Hemostasis was achieved. The skin\nwas closed and dressed in usual fashion. The patient tolerated the procedure well.\nHome instructions were given.\n',
   'page': 33,
   'vertices': [(0.08695652335882187, 0.2507936656475067),
    (0.8947700262069702, 0.25158730149269104),
    (0.8947700262069702, 0.37460318207740784),
    (0.08695652335882187, 0.3738095164299011)],
   'char_start': 57119,
   'char_end': 57724,
   'sequence_matcher_score': 1.0},
  {'text': 'Previous ultrasound study also indicated possible large lymph node in the axilla.\nPrebiopsy repeat imaging through the axillary region failed to identify any cortical\nthickening or hilar effacement. Axillary biopsy was not performed.\n',
   'page': 33,
   'vertices': [(0.08695652335882187, 0.39325398206710815),
    (0.9048519134521484, 0.39325398206710815),
    (0.9048519134521484, 0.43690475821495056),
    (0.08695652335882187, 0.43690475821495056)],
   'char_start': 57724,
   'char_end': 57959,
   'sequence_matcher_score': 1.0},
  {'text': 'ULTRASOUND\n',
   'page': 6,
   'vertices': [(0.08947700262069702, 0.2996031641960144),
    (0.18714556097984314, 0.2996031641960144),
    (0.18714556097984314, 0.3083333373069763),
    (0.08947700262069702, 0.3083333373069763)],
   'char_start': 8793,
   'char_end': 8805,
   'sequence_matcher_score': 1.0},
  {'text': 'Bilateral mammogram 7/14/23\nBREAST COMPOSITION:\n',
   'page': 5,
   'vertices': [(0.09010712057352066, 0.7039682269096375),
    (0.3264020085334778, 0.704365074634552),
    (0.3264020085334778, 0.7293650507926941),
    (0.09010712057352066, 0.7289682626724243)],
   'char_start': 7637,
   'char_end': 7686,
   'sequence_matcher_score': 1.0},
  {'text': 'BREAST, RIGHT, 7:00, 5 CM FN, CORE BIOPSIES: BENIGN BREAST TISSUE, NO\nTUMOR PRESENT.\n',
   'page': 8,
   'vertices': [(0.11153119057416916, 0.7138888835906982),
    (0.7145557403564453, 0.7138888835906982),
    (0.7145557403564453, 0.7376984357833862),
    (0.11153119057416916, 0.7376984357833862)],
   'char_start': 14027,
   'char_end': 14113,
   'sequence_matcher_score': 1.0},
  {'text': '(Link Unavailable) Show images for MRI-BREAST BIL W/WO CON\n',
   'page': 26,
   'vertices': [(0.06616257131099701, 0.20317460596561432),
    (0.5229993462562561, 0.20317460596561432),
    (0.5229993462562561, 0.2142857164144516),
    (0.06616257131099701, 0.2142857164144516)],
   'char_start': 45213,
   'char_end': 45273,
   'sequence_matcher_score': 1.0},
  {'text': 'MRI-BREAST BIL W/WO CON: Result Notes\n',
   'page': 26,
   'vertices': [(0.09010712057352066, 0.3301587402820587),
    (0.4883427917957306, 0.3301587402820587),
    (0.4883427917957306, 0.3432539701461792),
    (0.09010712057352066, 0.3432539701461792)],
   'char_start': 45415,
   'char_end': 45454,
   'sequence_matcher_score': 1.0},
  {'text': 'Study Result\nNarrative & Impression\nMRI-BREAST BIL W/WO CON\n',
   'page': 26,
   'vertices': [(0.06490232795476913, 0.6214285492897034),
    (0.30938878655433655, 0.6202380657196045),
    (0.3100188970565796, 0.6702380776405334),
    (0.06553245335817337, 0.6714285612106323)],
   'char_start': 45719,
   'char_end': 45780,
   'sequence_matcher_score': 1.0},
  {'text': 'TECHNIQUE: MRI-BREAST BIL W/WO CON. Multiplanar T1 and T2 weighted images were\nobtained prior to and post intravenous injection of Clariscan. The color-flow\ndynamics, maximum intensity projection and time intensity curves are reviewed. The\nimages were obtained using a dedicated breast coil on a 1.5 Tesla or greater magnet.\n',
   'page': 26,
   'vertices': [(0.08695652335882187, 0.7384920716285706),
    (0.9023314714431763, 0.7384920716285706),
    (0.9023314714431763, 0.7984126806259155),
    (0.08695652335882187, 0.7984126806259155)],
   'char_start': 45877,
   'char_end': 46203,
   'sequence_matcher_score': 1.0},
  {'text': 'R diagnostic mammogram/US:7/21/23\n',
   'page': 6,
   'vertices': [(0.0907372385263443, 0.07896825671195984),
    (0.4108380675315857, 0.07896825671195984),
    (0.4108380675315857, 0.09087301790714264),
    (0.0907372385263443, 0.09087301790714264)],
   'char_start': 8278,
   'char_end': 8313,
   'sequence_matcher_score': 1.0},
  {'text': 'TECHNIQUE: Right 2-D and 3-D tomosynthesis images were obtained. Computer aided\ndetection (CAD) was used to assist in the interpretation.\n',
   'page': 6,
   'vertices': [(0.08947700262069702, 0.12579365074634552),
    (0.8582230806350708, 0.12579365074634552),
    (0.8582230806350708, 0.15357142686843872),
    (0.08947700262069702, 0.15357142686843872)],
   'char_start': 8367,
   'char_end': 8506,
   'sequence_matcher_score': 0.8260869565217391},
  {'text': 'TECHNIQUE: 2-D and 3-D tomosynthesis images of the right breast were obtained in the\nCC, ML, and MLO projections in attempts to identify the previous mammographic mass of\n',
   'page': 56,
   'vertices': [(0.08632640540599823, 0.1726190447807312),
    (0.9149338603019714, 0.1726190447807312),
    (0.9149338603019714, 0.20079365372657776),
    (0.08632640540599823, 0.20079365372657776)],
   'char_start': 87289,
   'char_end': 87461,
   'sequence_matcher_score': 0.8260869565217391}]}
```

##  Key Decisions 



1. Full text prior to chunking
   - increase semantic value and coherency.
3. Text Chunking
     - Chunk and overlap size (1000, 100) -> long enough for meaningful content but not too long. Overlap to avoid misunderstandings.
     - Text Spliter(LangChain RecursiveCharacterTextSplitter) -> splits on paragraph and sentence boundaries before resorting to character count.
4. Vectorization -> `text-embedding-ada-002`
   - Older model but performs well.
   - Good benchmark.
5. Indexing
   - Faiss is fast and good for prototyping. In production, vector DB have built in solutions.
   - `IndexFlatIP`: flat index that computes the exact inner product between the query vector and all indexed vectors.
   - Given processing one document at a time, index that prioritizes precision (all indexed items are searched) over speed.
   - Vectors are normalized. This removes magnitude as a consideration. Makes the inner product equivalent to cosine similarity. Using cosine similarity gives a readily interpretable score ranging from -1 (completely dissimilar) to 1 (extremely similar).
6. Retrieval
   - index search
   - filter chunks by cosine score from search.
   - filtered chunks provided as context.
7. Answer Generation
   - LLM: `GPT-4o`. Good benchmark. Low rates of hallucination.
   - LLM returns the answers as well as grounding -> substrings used to answer the question.
   - Effectively, LLM performs the reranking.
   - Prompt Engineering theory: CoT, emotional stimuation, grounding.
   - Retrieve blocks containing the substrings using SequenceMatcher.
     





## Step-by-Step Description


In order to explain the process, I will go step by step through the `process_and_answer_question` in `submission.py`.

Here is a reduced version of that function with error handling and logging removed. You can see there are four main components: Parse pdf, Create full_text and blocking_mapping, long document quesiton answering and short document question answering. I will go through each of these steps in detail.


```python
def process_and_answer_question(
    pdf_path: str, question: str, k: int
) -> Dict[str, Any]:
    """
    Process a PDF file and answer a question based on its contents.
    """

    input_path = Path(pdf_path)
    
    #1. Parse pdf.
    parsed_document = await process_pdf(input_path)
    #2. Creat full_text and block_mapping
    full_text, block_mapping = create_full_text_and_mapping(parsed_document)
    #3. Long document question answering
    if len(full_text) >= SHORT_DOC_THRESHOLD:
        #4. Chunk text
        chunks_with_metadata = chunk_text_with_metadata(full_text, block_mapping)
        #5. Index text
        index, chunks_with_metadata = create_index(chunks_with_metadata)
        #6. Answer question.
        result = answer_question(index, chunks_with_metadata, question, k)
    #7. Short document question answering
    else:
        result = answer_question_short_doc(full_text, question, block_mapping)
    
    return result
```


### 1. Parse PDF

The first step is to process the PDF and extracting its content. This is done using the `process_pdf` function from the `pdf_processor.py` module. The majority of the code comes from `pdf.py`, which was supplied.



This function essentially creates an instance of DocumentParser and then uses it. Here is a reduced version of the function:
```python
def process_pdf(input_path: Path) -> ParsedDocument:
    """
    Process a PDF file and create a ParsedDocument object.

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

    except Exception as e:
        # If any error occurs during processing, raise a RuntimeError
        raise RuntimeError(f"Error processing PDF: {str(e)}")
```


The `DocumentParser` class (provided in the `pdf.py` file) is the main tool for this step. As this file was provided, I will keep comments short. Here are key points about how it functions:

- It uses Google Cloud Document AI for advanced OCR and document understanding.
- It processes PDFs in batches to handle large documents efficiently.
- It's designed to work asynchronously, allowing for better performance in scenarios where multiple documents need to be processed.
- The result is a structured representation of the document, with text blocks and their positions on each page.

This approach allows for high-quality text extraction and layout preservation, which is crucial for accurately answering questions about the document's content later in the process.

### 2. Create full text and mapping

Full text and block mapping creation is completed through the  `create_full_text_and_mapping` function. 


```python
#2. Creat full_text and block_mapping
full_text, block_mapping = create_full_text_and_mapping(parsed_document)
```



This function takes a ParsedDocument object as input and returns a tuple containing two elements: a string (the full text of the document) and a list of dictionaries (the block mapping).

Note that blocks can be quite small (individual words). It is essential for index searching to create larger, more meaningful chunks of texts. In order to chunk the text, it is necessary to create the full string.

This function performs the following steps:

    1. Iterates through all pages and blocks in the parsed document
    2. Concatenates all block texts into a single string
    3. Creates a mapping of each block's position in the full text
    4. Adjusts page numbers to be 1-indexed

It returns a tuple containing:

    - The full text of the document (str)
    - A list of dictionaries, each representing a block and its position in the full text

The position of each block within the text (character index from `full_string`) is used downstream to group the  blocks to the chunks of text.

```python
def create_full_text_and_mapping(parsed_document: ParsedDocument) -> Tuple[str, List[Dict]]:
    """
    Create full text and block mapping from a ParsedDocument.
    
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

    # Return the full text and the block mapping
    return full_text, block_mapping

```


### **3. Long Document Question Answering**
```python

    #3. Long document question answering
    if len(full_text) >= SHORT_DOC_THRESHOLD:
        ...
```
I opted to create two workflows, one for longer texts and one for shorter texts. The limit for this is stored as a variable, `SHORT_DOC_THRESHOLD`. The default value is 1000 characters. Meaning there must be at least two chunks in order to use long document processing. This could arguably be increased to closer to the token limit of the relevant model.

After generating `full_text`, we then pass to question answering. Question answering is completed through `answer_question`. This process is comprised of three steps: text chunking and mapping, vectorization, and indexing.



### **4. Chunk Text**
```python

        #4. Chunk text
        chunks_with_metadata = chunk_text_with_metadata(full_text, block_mapping)
       
```

Text chunking is accomplished via the function `chunk_text_with_metadata`. This function performs the following steps:

    1. Initialize a RecursiveCharacterTextSplitter with specific parameters.
    2. Split the input text into chunks.
    3. For each unique chunk:
       a. Generate a unique identifier.
       b. Find its position in the original text.
       c. Identify relevant blocks from the block_mapping that overlap with the chunk.
       d. Create a tuple of (chunk, relevant_blocks, chunk_id).
    4. Return a list of these tuples.







```python
def chunk_text_with_metadata(
    text: str, block_mapping: List[Dict]
) -> List[Tuple[str, List[Dict], str]]:
    """
    Split the given text into chunks and associate each chunk with its corresponding metadata.

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
```

##### **Key Design Choices**

**1. Chunk Size (1000 characters)**: This size provides a good balance between context and specificity as well as efficiency of processing. Smaller chunks might lack context, while larger chunks might introduce irrelevant information. This equates to approximately 140-250 words.

**2. Chunk Overlap (100 characters)**: This helps maintain continuity between chunks, reducing the chance of breaking up important phrases or sentences.

**3. Text Splitter (RecursiveCharacterTextSplitter)**: This LangChain splitter is chosen for its ability to respect document structure, adaptively splitting on paragraph and sentence boundaries before resorting to character count. This helps maintain the coherence of text chunks. RecursiveCharacterTextSplitter is designed to split text in a way that respects the structure of the document. It tries to split on paragraph breaks, then sentence breaks, and finally on character count if needed. This adaptive approach helps maintain the coherence of the text chunks. As with other text splitters, you can also configure chunk length and overlap.

#### **5. Create Index**
```python
        #5. Index text
        index = create_index(chunks_with_metadata)

```



Embeddings and the index are generated through the `create_index` function.

This function performs the following steps:

        1. Generates embeddings for each chunk of text
        2. Normalizes each embedding vector
        3. Creates a FAISS IndexFlatIP (Inner Product) index
        4. Adds the normalized embeddings to the index



``` python
def create_index(
    chunks_with_metadata: List[Tuple[str, List[Dict], str]]
) -> faiss.IndexFlatIP:
    """
    Create a FAISS index from the given chunks and their metadata.

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
```

##### **Embedding Model**
For simplicity, I chose `text-embedding-ada-002`. While older than other OpenAI models, I find it performs better. In order to determine a final model, it would be useful to review the relevant leaderboards such as https://huggingface.co/spaces/mteb/leaderboard, specifically the retrieval rankings. Again, finding deployments that are HIPAA compliant is paramount.

##### **Index**

For indexing, I chose Faiss IndexFlatIP. In terms of indexing, Faiss allows for fast similarity search, which is crucial when dealing with large documents or a large corpus of documents. In production, many databases offer vector databases along with indexing which could streamline the process.

IndexFlatIP is a flat index that computes the exact inner product between the query vector and all indexed vectors. Given that the task was one document at a time, I chose an index that prioritizes precision (all indexed items are searched) over speed. If the corpus size were larger, a different index could be more appropriate.

In addition, all vectors are normalized. Normalization removes magnitude as a consideration and makes the inner product equivalent to cosine similarity. Using cosine similarity gives a readily interpretable score ranging from -1 (completely dissimilar) to 1 (extremely similar).



#### **6. Answer Question**

```python
        #6. Answer question.
        result = answer_question(index, chunks_with_metadata, question, k)
```


Answers are generated through the `answer_question`.  This function takes five parameters: `index` (FAISS index for similarity search), `chunks_with_metadata` (text chunks with metadata), `question` (user's query), `k` (number of chunks to retrieve), and `similarity_threshold` (for filtering chunks). The pre-computed Faiss index quickly identifies relevant chunks of text from a large corpus. Providing `k` and `similarity_threshold` as parameters allows for searches to be modified to suit the task (question complexity, document length).



```python
def answer_question(
    index: faiss.IndexFlatIP,
    chunks_with_metadata: List[Tuple[str, List[Dict], str]],
    question: str,
    k: int,
    similarity_threshold: float = 0.5,
) -> Dict:
    """
    Answer a question based on the provided index and chunks.

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
    
    # Perform cosine similarity search using FAISS
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
```






##### **Key Design Choices:**

In this section, I detail key design choices. Some of these choices can be fine-tuned based on specific use cases and performance requirements. The system is designed to be flexible, allowing for adjustments to handle different question complexities and corpus sizes. 

**1. Generate and normalize the embedding for the input question.**
   - Generates and normalizes an embedding for the question.
   - The same emedding model is used for chunks and the question.

**2. Perform a similarity search using FAISS to find the k most similar chunks.**
   - Utilizes the Faiss index to efficiently find the `k` most similar chunks to the question.
   - The index returns the k most similar chunks based on the cosine similarity between the question embedding and the embeddings of the chunks in the index.
   - K is adjusted to be the min of k and the length of the index. Faiss will return k items regardless of index length. This step ensure no duplicates are returned from the index search.
   
**3. Filter and sort the retrieved chunks based on similarity scores.**
   - Applies `filter_chunks_by_similarity_cosine` to refine the selection based on the similarity threshold.
   
**4. Extract relevant chunks and their texts.**
- Create the context by processesing the relevant chunks.
   
**5. Generate an answer using GPT-4o, providing the relevant chunk texts as context.**

I've chosen to use OpenAI's API and GPT-4o to generate an answer. One advantage of using OpenAI models is that they are available through Azure as HIPAA-compliant. GPT-4o because it is a good benchmark. For final model selection, addtional research based on the given project would be necessary. Additionally, GPT-4o has a low rate of hallucination as per the Huggingface Leaderboard: https://huggingface.co/spaces/vectara/Hallucination-evaluation-leaderboard

**Prompt**

In terms of prompting, the system content provides instructions to act as a medical assistant, answering the question based on the provided context and identifying the specific substrings of the context it used to formulate its answer. For user content, it is provided with the question and retreived chunks.

**Theoretically Underpinnings of Prompt**
- Asking the LLM to cite its source substrings accomplishes reranking.
- The prompt includes the phrase "Let's think about this step by step." This approach, Zero-Shot Chain-of-Thought prompting, has been shown to improve LLM output. See [here](https://arxiv.org/pdf/2205.11916) and [here.](https://arxiv.org/abs/2201.11903)
- The prompt includes the phrase "Your answer is really important for the provider and patient." Emotional stimulation has been shown to improve LLM output. The researchers developed "EmotionPrompt," a method combining prompts with emotional stimuli to evaluate and enhance LLM performance. Inspired by psychological theories like self-monitoring, social cognitive theory, and cognitive emotion regulation, they designed 11 emotional stimuli, such as "You'd better be sure," and "Believe in your abilities and strive for excellence."  Overall, the researchers report “10.9% average improvement in terms of performance, truthfulness, and responsibility metrics” (2).  Second, responses generated with EmotionPrompt were more coherent, accurate, and ethically responsible. Overall, the human evaluation of the outpu validated EmotionPrompt's ability to enhance LLMs' generative capabilities, making their outputs more coherent, truthful, and responsible, with significant potential for improving AI-generated content quality in various applications. Essentially, an emotional appeal to the LLM improves LLM performance. See [here.](https://arxiv.org/abs/2307.11760)
- The identification of substring helps increase the accuracy of its response by requiring the LLM to provide self-grounding. The article cited here takes a different approach but contributed to my thinking on the topic. While the researchers trained a model using NLI (Natural Language Inference) to teach the LLM about supported/unsupported claims, the prompting technique for the trained model is similar to that which I used. During inference, the fine-tuned LLM is prompted to generate responses with citations and identify unsupported statements. In asking the LLM to provide the substrings used in its answer, I am asking for it to provide a citation and thus ensuring the answer is supported. See [here.](https://arxiv.org/html/2311.09533v2)




**6. Parse the generated answer and extract substrings used in the answer.**
   - Extracts the content of the OpenAI's response.
   - Parses the response as JSON / a Python dictionary.
   - Removes duplicates from the substrings list.
   - The response contains two parts: the answer and substrings as described above.
   

**7. Find relevant blocks for each substring within filtered chunks.**
- Via the function `find_blocks_with_substring`, identifies blocks that contain a substring.
- This function performs two types of searches:
    - Exact match: It first looks for blocks that contain the exact substring.
    - Similarity match: If no exact matches are found, it searches for blocks with text similar to the substring, using a similarity threshold.
- The similarity is calculated using the SequenceMatcher ratio. Defaults to 0.75.

**SequenceMatcher**
- SequenceMatcher ratio compares sequences of characters and returns a value between 0 and 1, where 1 means perfect similarity. 
- Matching process: 
    - It starts matching by finding the longest contiguous matching subsequence between the two strings. 
    - Then it recursively applies this process to the unmatched regions on both sides of this match. 
    - This continues until all matching subsequences are found.

**8. Remove duplicate blocks from the set of relevant blocks.**

**9. Sort the blocks by sequence_matcher_score.**
- If sorting by location is preferred, please see `sort_blocks`. 

**10. Return a dictionary containing the answer, substrings, and scored relevant blocks.**










### 4. Short document question answering
If the objective is 'chat with my PDF', it may be useful to have an alternataive workflow for shorter documents. 

```python

    #7. Short document question answering
    else:
        result = answer_question_short_doc(full_text, question, block_mapping)
```

Note that `SHORT_DOC_THRESHOLD` is defined in `submission.py`. If short document processing is to be retained, this could be supplied as an argument instead.

```python
# Define a threshold for short documents (e.g., 1000 characters, estimated to be 140-200 words)
# Documents shorter than this will be processed differently
SHORT_DOC_THRESHOLD = 1000
```

When the character length of `full_text` is less than the `SHORT_DOC_THRESHOLD`, question answering is completed through `answer_question_short_doc`. When a text is shorter, it is unnecessary to chunk, index, and retrieve relevant chunks. Instead, the full text is provided as context. The overall process is very simialr to `answer_question` without the index search. GPT is given the full text as context, provides substrings, and then all subsequent steps are the same. 




## Future Improvements
- Index: I would go directly to a cloud-based solution rather than local index (FAISS)
- Evaluation: I would add an evaluation step at the end of the process which could be used to trigger regeneration of the response based on scores.
- Store prompts as YAML.
- Consider preferences for arguments (question answering taking in PDF or parsed PDF).
- Currently, the output files store as the same name as the input file with the extension changed. In the assignemnt, 'output' was mentioned as a file name. If this is preferred, this can easily be changed.



