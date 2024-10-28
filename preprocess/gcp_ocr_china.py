from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor  # Add this import

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import pandas as pd
import os
from tqdm import tqdm

def create_processor(
    project_id: str, location: str, processor_display_name: str
) -> documentai.Processor:
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    # The full resource name of the location
    # e.g.: projects/project_id/locations/location
    parent = client.common_location_path(project_id, location)

    # Create a processor
    return client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            display_name=processor_display_name, type_="OCR_PROCESSOR"
        ),
    )

def process_document(processor_name: str, file_path: str) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(
        content=image_content, mime_type="application/pdf"
    )

    # Configure the process request
    request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)

    result = client.process_document(request=request)

    return result.document

def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document"s text. This function converts
    offsets to a string.
    """
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in layout.text_anchor.text_segments
    )

def pdf_processor(processor_name: str, extracted_data, docs_path: str) -> list[dict]:
# Loop through each PDF file in the "docai" directory.
    for path in tqdm(glob.glob(f"{docs_path}/*.pdf")):
        # Extract the file name and type from the path.
        file_name, file_type = os.path.splitext(path)
            
        # Process the document.
        document = process_document(processor_name, file_path=path)

        if not document:
            print("Processing did not complete successfully.")
            continue

        # Split the text into chunks based on paragraphs.
        document_chunks = [
            layout_to_text(paragraph.layout, document.text)
            for page in document.pages
            for paragraph in page.paragraphs
        ]


        # Loop through each chunk and create a dictionary with metadata and content.
        for chunk_number, chunk_content in enumerate(document_chunks, start=1):
            # Append the chunk information to the extracted_data list.
            extracted_data.append(
                {
                    "file_name": file_name,
                    "file_type": file_type,
                    "chunk_number": chunk_number,
                    "content": chunk_content,
                }
            )
    return extracted_data


if __name__ == "__main__":
    project_id = "iglintdb"
    location = "us"
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    
    save_dir = 'annual_csvs'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    split_dir = 'annual_splits'
    countries = os.listdir(split_dir)

    def process_company(company, country):
        if country != "China":  # Only process if the country is China
            return  # Skip processing for other countries
        processor_display_name = f"{company}-ocr-processor-1"  # Unique processor name for each company
        processor = create_processor(project_id, location, processor_display_name)
        processor_name = processor.name
        extracted_data = []
        
        years = os.listdir(os.path.join(split_dir, country, company))
        for year in years:                
            sub_dir = os.path.join(save_dir, country, company, year)
            if not os.path.exists(sub_dir):
                docs_path = os.path.join(split_dir, country, company, year)
                extracted_data = pdf_processor(processor_name, extracted_data, docs_path)

                # Convert extracted_data to a sorted Pandas DataFrame
                pdf_data = (
                    pd.DataFrame.from_dict(extracted_data)
                    .sort_values(by=["file_name"])
                    .reset_index(drop=True)
                )
                os.makedirs(sub_dir)
                save_path = os.path.join(sub_dir, "results.csv")
                pdf_data.to_csv(save_path, index=False)

    def process_company_wrapper(company_country_tuple):
        company, country = company_country_tuple
        return process_company(company, country)

    with ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor for parallel processing
        for country in countries:
            companies = os.listdir(os.path.join(split_dir, country))
            # Create a list of tuples for company and country
            company_country_tuples = [(company, country) for company in companies]
            # Wrap the companies list with tqdm for progress tracking
            list(tqdm(executor.map(process_company_wrapper, company_country_tuples), total=len(companies), desc=f"Processing companies in {country}"))
