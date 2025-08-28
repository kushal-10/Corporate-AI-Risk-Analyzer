# Corporate-AI-Risk-Analyzer

Toolkit for extracting AI-related content from corporate annual reports and measuring associated risks.

## Features

- **Preprocess reports** – split PDFs, gather metadata, run OCR and translate text so that every company/year pair has a plain‑text file for analysis.
- **Retrieve AI passages** – scan the text for AI/ML keywords and gather surrounding context into a searchable JSON index.
- **Classify sentiment** – label each extracted passage as POSITIVE or NEGATIVE using DistilBERT, RoBERTa, or GPT classifiers.
- **Quantify risk** – aggregate passage labels into document‑level risk scores and compile results by country, sector, and revenue.
- **Visualize trends** – plot risk trends over time to compare regions, industries, and company sizes.

## Setup

```bash
pip install -r requirements.txt
```

## Preprocess

Run the following scripts in order to prepare annual reports:

```bash
python preprocess/clean_pdfs.py     # split PDF reports and extract company metadata
python preprocess/gcp_ocr.py        # OCR pages with Google Document AI
python preprocess/translate.py      # merge OCR output and translate non‑English text
python preprocess/check_txts.py     # verify that all text files are present
```

The steps above produce structured text under `annual_txts/` and a metadata file `preprocess/company_metadata.json`.

## Retrieval and classification

Extract AI‑related passages and assign sentiment labels:

```bash
python retrieval/regex_retrieval.py       # build results/retrieved_docs.json
python retrieval/distillbert_scores.py    # add labels using DistilBERT
python retrieval/roberta_scores.py        # add labels using RoBERTa
# optional
python retrieval/gpt_scores_a.py          # label passages with GPT‑4o‑mini
```

## Risk analysis

Use the labeled passages to compute and visualise risk metrics:

```bash
python risk/calculate_risk.py   # create risk/*.csv with aggregate risk scores
python risk/analysis.py         # generate plots in the plots/ directory
```

## Results

Generated artefacts such as retrieved passages, model labels, risk tables, and plots live under the `results/`, `risk/`, and `plots/` directories.

