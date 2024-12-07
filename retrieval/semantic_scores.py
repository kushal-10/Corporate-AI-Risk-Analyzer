"""
DEPRECATED
"""

import json
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# Function to compute semantic scores
def compute_semantic_scores(text, model, tokenizer):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    return scores


if __name__ == "__main__":
    # Load existing documents from JSON file
    with open('retrieval/retrieved_docs.json', 'r') as json_file:
        docs_dict = json.load(json_file)  # Load existing data

    # Load existing semantic scores if they exist
    sem_scores_dict = {}
    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Process each document and compute semantic scores
    doc_keys = list(docs_dict.keys())
    for key in tqdm(doc_keys, desc="Processing documents for semantic scores"):
        if key not in sem_scores_dict:
            sem_scores_dict[key] = []
            texts = docs_dict[key]
            for text in texts:
                scores = compute_semantic_scores(text, model, tokenizer)
                sem_scores_dict[key].append(scores.tolist())  # Convert ndarray to list

    with open('retrieval/semantic_scores.json', 'w') as json_file:
        json.dump(sem_scores_dict, json_file, indent=4)