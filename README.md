# business-insights

## Preprocess
`clean_pdfs.py` - Cleans the pdfs and extracts company metadata. 

`company_metadata.json` - Contains the company metadata.

```bash
python preprocess/clean_pdfs.py
```
Run this script to clean the pdfs, split them into chunks and extract company metadata.


`gcp_ocr.py` - Uses Google Cloud OCR to extract text from the pdfs.

```bash
python preprocess/gcp_ocr.py
```

This script processes the pdfs and saves the results in `annual_csvs`.


## Retrieval

