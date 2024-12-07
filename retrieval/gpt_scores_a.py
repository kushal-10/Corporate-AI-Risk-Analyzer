from openai import OpenAI
import os
import json
from tqdm import tqdm

client = OpenAI()

BASE_PROMPT = """
Assess the SENTENCE provided and classify its stance on Artificial Intelligence (AI) as POSITIVE or NEGATIVE:  
- Choose POSITIVE if the statement suggests rewards, potential, or favorable outcomes of AI.  
- Choose NEGATIVE if it identifies risks, limitations, or neutral ideas with no clear opportunity.  

SENTENCE:  
"""

END_PROMPT = """
Provide only one-word: POSITIVE or NEGATIVE.  
"""

def classify(input_prompt: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": input_prompt + END_PROMPT
            }
        ]
    )

    return completion.choices[0].message.content

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
    output_path = os.path.join('results', 'gpt_labels_a.json')
    
    get_labels(json_path, output_path)


