import json
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
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


MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

# Function to compute semantic scores
def compute_semantic_scores(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
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
    try:
        with open('retrieval/semantic_scores.json', 'r') as json_file:
            sem_scores_dict = json.load(json_file)  # Load existing semantic scores
    except FileNotFoundError:
        pass  # If the file doesn't exist, start with an empty dictionary

    # Process each document and compute semantic scores
    doc_keys = list(docs_dict.keys())
    for key in tqdm(doc_keys, desc="Processing documents for semantic scores"):
        if key not in sem_scores_dict:
            sem_scores_dict[key] = []
            texts = docs_dict[key]
            for text in texts:
                scores = compute_semantic_scores(text)
                sem_scores_dict[key].append(scores)

    with open('retrieval/semantic_scores.json', 'w') as json_file:
        json.dump(sem_scores_dict, json_file, indent=4)