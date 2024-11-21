"""
Check missings text files
"""

import os

## Find missing text files in annual_txts
base_dir = "annual_txts"
companies = os.listdir(base_dir)
for company in companies:
    years = os.listdir(os.path.join(base_dir, company))
    for year in years:
        txt_path = os.path.join(base_dir, company, year, "results.txt")
        if not os.path.exists(txt_path):
            print(f"Missing {txt_path}")


## Find missing extractions in annual_txts
base_dir = "annual_splits"
companies = os.listdir(base_dir)
for company in companies:
    years = os.listdir(os.path.join(base_dir, company))
    for year in years:
        pth1 = os.path.join(base_dir, company, year)
        pth2 = os.path.join('annual_txts', company, year)
        if os.path.exists(pth1) and not os.path.exists(pth2):
            print(f"Missing {pth2}")
