import os
import json
import pandas as pd
from tqdm import tqdm
import riskv2.risks as RISK

# with open(os.path.join('retrieval', 'labels.json'), 'r') as f:
#     label_data = json.load(f)

# with open(os.path.join('retrieval', 'gpt_labels.json')) as f:
#     gpt_data = json.load(f)

def get_metadata(data_path: str):
    splits = data_path.split('_')
    company_name = splits[0].split('.')[-1]
    revenue = splits[1].replace('$', '').replace(' ', '').strip().replace('B', '')
    if 'T' in revenue:
        revenue = revenue.replace('T', '')
        revenue = float(revenue) * 1000
    else:
        revenue = float(revenue)
    sector = splits[2].strip()

    # Clean up sectors
    if sector == 'Consumer Staplers' or sector == 'Consumer Stapler':
        sector = 'Consumer Staples'
    elif sector == 'FInancial Service' or sector == 'Financials':
        sector = 'Financial Service'
    elif sector == 'Information Tech':
        sector = 'Information Technology'
    elif sector == 'Industrials':
        sector = 'Industries'

    return company_name, sector, revenue


def generate_dataframe(json_data: str, output_path: str = "risk_data.csv"):
    keys = list(json_data.keys())

    countries = []
    firms = []
    sectors = []
    revenues = []
    years = []
    risks = []

    for k in tqdm(keys):
        results = json_data[k]
        # risk = RISK.naive_risk(results)
        risk = RISK.np_risk(results)
        if risk:
            risks.append(risk)
            splits = k.split('/')
            countries.append(splits[0])
            data_path = splits[1]
            years.append(int(splits[2]))
            company, sector, revenue = get_metadata(data_path)
            firms.append(company)
            sectors.append(sector)
            revenues.append(revenue)
        

    # Create DataFrame
    df = pd.DataFrame({
        'country': countries,
        'firm': firms,
        'sector': sectors,
        'revenue': revenues,
        'year': years,
        'risk': risks
    })

    # Sort DataFrame
    df = df.sort_values(['country', 'firm', 'year'])

    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == '__main__':

    file_mapping = {
        "distillbert_labels.json": "distillbert.csv",
        "roberta_labels.json": "roberta.csv",
        "gpt_labels_a.json": "gpta.csv",
        "gpt_labels_b.json": "gptb.csv",
        "gpt_labels_c.json": "gptc.csv"
    }

    for k in list(file_mapping.keys()):

        with open(os.path.join('results', k), 'r') as f:
            label_data = json.load(f)

        generate_dataframe(label_data, os.path.join('v2resuts', file_mapping[k]))
   