import os
import json
from tqdm import tqdm
import re
from typing import List, Tuple

# Define the directory path at the top of the file
directory_path = "annual_txts"

def split_into_chunks(text: str, chunk_size: int = 250, overlap: int = 25) -> List[str]:
    """Split text into chunks of specified size with overlap."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        # Get chunk_size words
        end = start + chunk_size
        chunk = ' '.join(words[start:min(end, len(words))])
        chunks.append(chunk)
        
        # Move start position, considering overlap
        start = end - overlap
    
    return chunks

def find_keyword_in_passage(passage: str) -> Tuple[bool, str]:
    """Check if any keyword exists in the passage and return the matched term."""
    keywords = [
        r'artificial intelligence',
        r'machine learning',
        r'deep learning',
        r'neural network',
        r'natural language processing',
        r'computer vision',
        r'robotics',
        r'data mining',
        r'big data'
    ]
    
    passage_lower = passage.lower()
    for pattern in keywords:
        match = re.search(pattern, passage_lower)
        if match:
            return True, match.group()
    return False, ""

def extract_passages_around_keyword(text: str) -> List[Tuple[str, str]]:
    """
    First split text into overlapping chunks, then find keywords in each chunk.
    Returns list of tuples (matched_term, context).
    """
    # Split into chunks
    chunks = split_into_chunks(text)
    
    matches = []
    for chunk in chunks:
        has_keyword, matched_term = find_keyword_in_passage(chunk)
        if has_keyword:
            matches.append((matched_term, chunk))
    
    return matches

def extract_semantic_passages_with_context(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return extract_passages_around_keyword(text)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def extract_passages(output_file: str):
    json_metadata = {}

    # Check if the output file exists and load existing data if it does
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            json_metadata = json.load(f)

    countries = ['USA', 'India', 'China', 'Germany']
    for country in countries:
        firms = os.listdir(os.path.join(directory_path, country))
        for firm in firms:
            years = os.listdir(os.path.join(directory_path, country, firm))
            for year in years:
                json_key = os.path.join(country, firm, year)
                txt_path = os.path.join(directory_path, country, firm, year, 'results.txt')
                # Check if the key already exists
                if json_key not in [list(json_metadata.keys())]:
                    extracted_passages = extract_semantic_passages_with_context(txt_path)
                    json_metadata[json_key] = extracted_passages
                    if len(extracted_passages) == 0:
                        print(f"No extracted passages for : {json_key}")

                    # Save the updated metadata to the output file after processing each year
                    with open(output_file, 'w') as f:
                        json.dump(json_metadata, f, indent=4)


if __name__ == '__main__':
    output_file = os.path.join("retrieval", "retrieved_docs.json")
    extract_passages(output_file)
    print(f"Retrieved Docs saved to : {output_file}")