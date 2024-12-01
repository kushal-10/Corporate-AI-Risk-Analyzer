import os
import re
from tqdm import tqdm
from typing import List
import unicodedata

def clean_text(text: str) -> str:
    """Clean text by removing unwanted characters and normalizing unicode."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove zero-width spaces and other invisible unicode characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202f\ufeff]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
    
    return text.strip()

def merge_lines(lines: List[str]) -> str:
    """Merge lines into a single passage with proper spacing."""
    # Remove empty lines and clean each line
    cleaned_lines = [clean_text(line) for line in lines if line.strip()]
    
    # Join with space, ensuring proper sentence spacing
    text = ' '.join(cleaned_lines)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'(\w)\s+\'(\w)', r"\1'\2", text)  # Fix contractions
    
    return text

def process_files():
    base_dir = "annual_txts"
    
    # Get valid countries (directories only, no hidden files)
    countries = [f for f in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    
    for country in tqdm(countries, desc="Processing Countries"):
        country_path = os.path.join(base_dir, country)
        firms = [f for f in os.listdir(country_path) 
                if os.path.isdir(os.path.join(country_path, f)) and not f.startswith('.')]
        
        for firm in tqdm(firms, desc=f"Processing {country} Firms", leave=False):
            firm_path = os.path.join(country_path, firm)
            years = [y for y in os.listdir(firm_path) 
                    if os.path.isdir(os.path.join(firm_path, y)) and not y.startswith('.')]
            
            for year in years:
                txt_path = os.path.join(firm_path, year, 'results.txt')
                if not os.path.exists(txt_path):
                    continue
                
                # Read and process the file
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Merge and clean the text
                    cleaned_text = merge_lines(lines)
                    
                    # Write back to the same file
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                        
                except Exception as e:
                    print(f"Error processing {txt_path}: {str(e)}")

if __name__ == "__main__":
    process_files()
