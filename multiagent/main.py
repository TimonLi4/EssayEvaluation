import pandas as pd
import os
from dotenv import load_dotenv
import tiktoken
from agents import MAS, FeedBack_stat_criteria, Feedback_fromLLM
from relevance import Relevance, Create_LDA_essay,clean

import fitz
import time

load_dotenv()


DATA_PATH = os.getenv('DATA_PATH')
file_path = os.getenv('FILE_PATH')
file_essay = os.getenv('file_essay')
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
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)
    text = ''
    final_score = 0
    dictionary, ldamodel = Create_LDA_essay(file_essay)
    

    text = extract_text_from_pdf(file_path)
    # text = df.iloc[0]['essay']

    # text = df.iloc[1]['text']
    print(text)

    new_text_clean = clean(text)
    new_text_tokens = new_text_clean.split()  # Преобразуем строку в список токен

    # Преобразуем новый текст в bag-of-words
    new_bow = dictionary.doc2bow(new_text_tokens)

    # Получаем распределение тем для нового текста
    new_text_topics = ldamodel.get_document_topics(new_bow)



    mas = MAS(text)
    results = mas.evaluate()
    
    statistic = pd.DataFrame(results)
    print(statistic)
    
    for result in results:
        grade = result['Grade']

        #######################
        if isinstance(grade, str) and '/' in grade:  
            grade = grade.split('/')[0]  

        final_score += float(grade) * result['Weights']
        
    
    with open('output.txt','w',encoding="utf-8") as file:
        file.write(f'Relevance: {Relevance(new_text_topics)}\n\n\n')
        file.write(statistic.to_string(index=False))
        file.write(FeedBack_stat_criteria(statistic))
        file.write(Feedback_fromLLM(statistic))
        file.write(f'Final score: {round(final_score,2)}')


    print('Final score: ',round(final_score,2))
    print(time.time()-start_time)