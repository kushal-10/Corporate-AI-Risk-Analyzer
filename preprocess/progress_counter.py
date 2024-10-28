import os

root_dir = 'annual_csvs'
countries = os.listdir(root_dir)

year_counter = 0
for country in countries:
    companies = os.listdir(os.path.join(root_dir, country))
    for company in companies:
        years = os.listdir(os.path.join(root_dir, country, company))
        for year in years:
            year_counter += 1

print(f"Total number of years processed: {year_counter*100/800}%")