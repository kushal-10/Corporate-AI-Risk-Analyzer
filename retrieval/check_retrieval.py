import json
import os 

with open(os.path.join('results', 'retrieved_docs.json')) as f:
    json_data = json.load(f)

counter = 0
for k in list(json_data.keys()):
    list_obj = json_data[k]
    counter += len(list_obj)

print(counter)