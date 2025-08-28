import json
import pandas as pd
from tqdm import tqdm
countries = []
companies = []
sectors = []
revenues = []
risks = []
years = []

def calculate_subrisk(sem_scores):
    neg = sem_scores[0]
    neu = sem_scores[1]
    pos = sem_scores[2]

    # risk_legacy = (neg - pos)/(1+neu)
    # normalized_risk = (1+risk_legacy)/2
    risk = neg+neu

    return risk


with open('retrieval/semantic_scores.json', 'r') as json_file:
    sem_scores_dict = json.load(json_file)  # Load existing semantic scores

file_locs = list(sem_scores_dict.keys())
for file_loc in tqdm(file_locs, desc="Processing documents for semantic scores"):

    splits = file_loc.split('/')
    year = splits[-2]
    country = splits[-4]

    splits = splits[-3].split('_')
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

    countries.append(country)
    companies.append(company_name)
    sectors.append(sector)
    revenues.append(revenue)

    scores = sem_scores_dict[file_loc]
    avg_risk = 0
    for s in scores:
        avg_risk += calculate_subrisk(s)
    avg_risk /= len(scores)
    risks.append(avg_risk)
    years.append(year)

df = pd.DataFrame({
    'country': countries,
    'company': companies,
    'sector': sectors,
    'revenue': revenues,
    'risk_legacy': risks,
    'year': years
})

df.to_csv('risk_legacy/risk_data.csv', index=False)
    
