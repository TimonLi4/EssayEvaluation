import pandas as pd
import os
from dotenv import load_dotenv
import tiktoken
from agents import MAS, FeedBack, output_fromLLM, text_for_feedback_only_stat_criteria

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
    # text = df.iloc[0]['essay']
    print(text)
    mas = MAS(text)
    results = mas.evaluate()
    
    statistic = pd.DataFrame(results)
    print(statistic)
    
    # print(FeedBack(text_for_feedback_only_stat_criteria(statistic)))
    # print(output_fromLLM(statistic))
    
    
    for result in results:
        grade = result['Grade']

        if isinstance(grade, str) and '/' in grade:  
            grade = grade.split('/')[0]  

        final_score += float(grade) * result['Weights']
        
    
    with open('output.txt','w') as file:

        file.write(statistic.to_string(index=False))
        file.write(FeedBack(text_for_feedback_only_stat_criteria(statistic)))
        file.write(output_fromLLM(statistic))
        file.write(f'Final score: {round(final_score,2)}')


    print('Final score: ',round(final_score,2))