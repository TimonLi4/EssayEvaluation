import pandas as pd
import os
from dotenv import load_dotenv
import tiktoken
from agents import MAS

import fitz

load_dotenv()


DATA_PATH = os.getenv('DATA_PATH')
file_path = os.getenv('FILE_PATH')
encoding = tiktoken.encoding_for_model('gpt-4')


# key_api = os.getenv('API_KEY')

def extract_text_from_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    text=''

    for page_num in range(pdf_doc.page_count):
        page_text=pdf_doc.load_page(page_num)
        text+=page_text.get_text()

    return text


if __name__ =='__main__':
    df = pd.read_csv(DATA_PATH)
    text = ''
    final_score = 0
    

    text = extract_text_from_pdf(file_path)
    print(text)
    mas = MAS(text)
    results = mas.evaluate()
    for result in results:
        print(result)
        final_score+=float(result['Grade'])*result['Weights']
    
    print('Final score: ',round(final_score,2))