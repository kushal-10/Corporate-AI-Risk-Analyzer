from pypdf import PdfReader, PdfWriter
import os 
import json
from tqdm import tqdm


def split_pdf_into_chunks(input_pdf_path, output_dir, chunk_size=10):
    # Open the PDF file

    with open(input_pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        
        # Loop over the pages and split them into chunks of 'chunk_size'
        for i in range(0, total_pages, chunk_size):
            pdf_writer = PdfWriter()
            
            # Add pages to the new PDF
            for j in range(i, min(i + chunk_size, total_pages)):
                pdf_writer.add_page(pdf_reader.pages[j])
            
            # Save the chunked PDF with a specific name
            chunk_number = (i // chunk_size) + 1
            output_filename = f'{chunk_number}.pdf'

            output_filename = os.path.join(output_dir, output_filename)
            with open(output_filename, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)

def split_pdfs():
    output_folder = 'annual_splits'
    countires = os.listdir('annual_reports')
    for country in countires:
        os.makedirs(os.path.join(output_folder, country), exist_ok=True)
        companies = os.listdir(f'annual_reports/{country}')
        for company in tqdm(companies, desc=f'Processing {country}'):
            os.makedirs(os.path.join(output_folder, country, company), exist_ok=True)
            pdfs = os.listdir(f'annual_reports/{country}/{company}')
            year_list = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
            for pdf in pdfs:
                if pdf.endswith('.pdf') or pdf.endswith('.PDF'):
                    for y in year_list:
                        if y in pdf:
                            save_dir = os.path.join(output_folder, country, company, y)
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir, exist_ok=True)
                                split_pdf_into_chunks(os.path.join('annual_reports', country, company, pdf), save_dir, chunk_size=10)
                            else:
                                print(f'{save_dir} already exists')
                
def get_company_metadata():
    company_metadata = []
    countires = os.listdir('annual_reports')
    for country in countires:
        pdfs = os.listdir(f'annual_reports/{country}')
        for pdf in pdfs:
            splits = pdf.split('_')
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

            company_metadata.append({
                'company_name': company_name,
                'revenue': revenue,
                'sector': sector,
                'country': country
            })


    with open(os.path.join('preprocess', 'company_metadata.json'), 'w') as f:
        json.dump(company_metadata, f, indent=4)

def check_years():
    countires = os.listdir('annual_reports')
    for country in countires:
        companies = os.listdir(f'annual_reports/{country}')
        for company in companies:
            pdfs = os.listdir(f'annual_reports/{country}/{company}')
            years = set()
            year_list = ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
            for pdf in pdfs:
                # Check for duplicate years
                for year in year_list:
                    if year in pdf:
                        if year in years:
                            print(f"{company} has multiple years")
                        else:   
                            years.add(year)          
        

if __name__ == '__main__':
    # get_company_metadata()
    # check_years()
    split_pdfs() 
    # pass    

#note - Microsoft is already in docx