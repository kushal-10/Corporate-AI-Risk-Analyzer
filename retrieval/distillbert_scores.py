import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


def classify(input_str: str, model=model, tokenizer=tokenizer):
    inputs = tokenizer(input_str, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]
    
    return label


if __name__=='__main__':

    test_str = "Artificial Intelligence"
    label = classify(test_str)

    print(label)

