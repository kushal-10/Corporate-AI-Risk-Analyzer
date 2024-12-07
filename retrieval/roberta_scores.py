# # Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier
from transformers import pipeline
import torch
import json
import os
from tqdm import tqdm


sentiment_classifier = pipeline("text-classification", model="Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier", device=torch.device("mps"))

def classify(input_str: str):
    sentinement_result = sentiment_classifier(input_str, max_length=512, truncation=True)
    label = sentinement_result[0]['label']
    label = label.upper()

    return label

def get_labels(json_path: str, output_path: str):
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    if not os.path.exists(output_path):
        label_metadata = {}
    else:
        with open(output_path, 'r') as f:
            label_metadata = json.load(f)

    keys = list(json_data.keys())
    processed_keys = list(label_metadata.keys())

    for k in tqdm(keys):
        if k not in processed_keys:
            passages = json_data[k]
            labels = []
            for p in passages:
                text_content = p[1]
                label = classify(text_content)
                labels.append([label, text_content])
            
            label_metadata[k] = labels
            with open(output_path, 'w') as f:
                json.dump(label_metadata, f, indent=4)


if __name__=='__main__':

    json_path = os.path.join('results', 'retrieved_docs.json')
    output_path = os.path.join('results', 'roberta_labels.json')
    
    get_labels(json_path, output_path)


