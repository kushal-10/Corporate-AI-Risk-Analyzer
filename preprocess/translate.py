## Check the language of the text, if it's not English, translate it.

import os
import pandas as pd
from langdetect import detect
from tqdm import tqdm
from transformers import AutoTokenizer, MarianMTModel

root_dir = 'annual_csvs'
save_dir = 'annual_txts'

def translate_text(text):
    src = "fr"  # source language
    trg = "en"  # target language

    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_text = text
    batch = tokenizer([sample_text], return_tensors="pt")

    generated_ids = model.generate(**batch)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# for country in os.listdir(root_dir):
#     for company in tqdm(os.listdir(os.path.join(root_dir, country)), desc=f"Processing {country}"):
#         for year in os.listdir(os.path.join(root_dir, country, company)):
#             csv_path = os.path.join(root_dir, country, company, year, f"results.csv")
#             save_subdir = os.path.join(save_dir, country, company, year)
#             if not os.path.exists(save_subdir):
#                 df = pd.read_csv(csv_path)
#                 text = ""
#                 for i in range(len(df)):
#                     text += df.iloc[i]['content']
#                 lang = detect(text)
#                 if lang == 'en':
#                     os.makedirs(save_subdir)
#                     save_path = os.path.join(save_subdir, f"results.txt")
#                     with open(save_path, 'w') as f:
#                         f.write(text)

translate_text("où est l'arrêt de bus ?")

