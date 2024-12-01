import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
from tqdm import tqdm

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


def classify(input_str: str, model=model, tokenizer=tokenizer):
    inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]
    
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

    json_path = os.path.join('retrieval', 'retrieved_docs.json')
    output_path = os.path.join('retrieval', 'labels.json')
    
    get_labels(json_path, output_path)


