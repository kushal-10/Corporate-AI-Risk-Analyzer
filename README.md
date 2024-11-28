# business-insights

## Setup

```bash
pip install -r requirements.txt
```

## Preprocess
`clean_pdfs.py` - Cleans the pdfs, checks for irregular entries and extracts company metadata. 

`company_metadata.json` - Contains the company metadata - `company_name`, `revenue`, `sector`, `country`.

```bash
python preprocess/clean_pdfs.py
```
Run this script to clean the pdfs, split them into chunks and extract company metadata. This will create a folder `annaul_splits` that takes in Annual Report PDFs and creates a split of 20 pages for each PDF and also saves the company metadata.      


`gcp_ocr.py` - Uses Google Cloud OCR to extract text from the pdfs. Refer [Document AI](https://cloud.google.com/document-ai?gad_source=1&gclid=CjwKCAiAxea5BhBeEiwAh4t5KzAGt23GzzyjNIASr8X2QW3Exe-hAFidSM4tBfP-MIz_L_3WN7o--RoCoeUQAvD_BwE&gclsrc=aw.ds&hl=en) for more details.


```bash
python preprocess/gcp_ocr.py
```

This script processes the pdfs and saves the results in `annual_csvs`.

```bash
python preprocess/translate.py
```

This script combines the extracted chunks from OCR and translates the text from Chinese/Hindi/German to English if required. All the translated text is saved in `annual_txts` - `country/company/year/results.txt`. Also available - [annualtxts](https://huggingface.co/Koshti10/annualtxts/tree/main)

```bash
python preprocess/check_txts.py
```

This script checks for missing text files in `annual_txts`.

## Retrieval
Run the following script to extract all passages that are relevant to the prompt - `Artificial Intelligence and Related Technologies`. Uses `all-MiniLM-L6-v2` for retrieval.

```bash
python retrieval/minilm_retrieval.py
```

Adjust the above prompt as required.


