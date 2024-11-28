import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Define the directory containing text files and your query
directory_path = "annual_txts"
query = "artificial intelligence and related technologies"

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the query
query_embedding = model.encode(query, convert_to_tensor=True)

def extract_semantic_passages_with_context(file_path, query_embedding=query_embedding, model=model, threshold=0.5, context_window=10):
    extracted = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            line_embedding = model.encode(line.strip(), convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, line_embedding).item()
            if similarity >= threshold:
                # Capture surrounding lines for context
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                context = lines[start:end]
                extracted.append((similarity, context))
    return extracted


def extract_passages(output_file: str):
    json_metadata = {}

    # Check if the output file exists and load existing data if it does
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            json_metadata = json.load(f)

    countries = os.listdir(os.path.join(directory_path))
    for country in countries:
        firms = os.listdir(os.path.join(directory_path, country))
        for firm in tqdm(firms, desc=f"Processing Firms for {country}"):
            years = os.listdir(os.path.join(directory_path, country, firm))
            for year in years:
                json_key = os.path.join(country, firm, year)
                txt_path = os.path.join(directory_path, country, firm, year, 'results.txt')
                # Check if the key already exists
                if json_key not in [list(json_metadata.keys())]:
                    extracted_passages = extract_semantic_passages_with_context(txt_path)
                    json_metadata[json_key] = extracted_passages
                    print(json_metadata[json_key] )

                    # Save the updated metadata to the output file after processing each year
                    with open(output_file, 'w') as f:
                        json.dump(json_metadata, f, indent=4)


if __name__ == '__main__':
    output_file = os.path.join("retrieval", "retrieved_docs.json")
    extract_passages(output_file)
    print(f"Retrieved Docs saved to : {output_file}")