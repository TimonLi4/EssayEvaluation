from django.shortcuts import render

import fitz
import pandas as pd
from dotenv import load_dotenv
import os

from .agents import MAS,Feedback_fromLLM,FeedBack_stat_criteria,Relevance
from .relevance import Score_all#,Create_LDA_article,Create_LDA_essay

load_dotenv()

file_essay = os.getenv('file_essay')
file_article = os.getenv('FILE_ARTICLE')


# file_essay = r'C:\Users\Timon4\Desktop\projectTrainee\AutomaticEssayEvaluation\media\uploads\essays.csv'
# file_article = r'C:\Users\Timon4\Desktop\projectTrainee\AutomaticEssayEvaluation\media\uploads\bbc_news_text_complexity_summarization.csv'

def main_page(request):
    return render(request,'EssayEvaluation/index.html')


def extract_text_from_pdf(uploaded_file):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
    text = ''
    for page in pdf_doc:
        text += page.get_text()
    return text


def relevance(text):

    value1,value2 = Score_all(text,file_essay,file_article)

    print(file_essay)
    print(file_article)
    result = ''

    if value1>value2:
        # print('final -- article')
        result = 'article'
    else:
        print('final -- essay')
        result = 'essay'
    return result


def upload_file(request):
    text = ''
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        text = extract_text_from_pdf(uploaded_file)

    final_score = 0
    mas = MAS(text)
    results = mas.evaluate()

    statistic = pd.DataFrame(results)

    feedback_stat_criteria = FeedBack_stat_criteria(statistic)
    feedback_LLM = Feedback_fromLLM(statistic)

    for result in results:
        grade = result['Grade']

        #######################
        if isinstance(grade, str) and '/' in grade:  
            grade = grade.split('/')[0]  

        final_score += float(grade) * result['Weights']

    relevance_LLM = Relevance(text)
    
    statistic = statistic[["name", 'Grade']]
    statistic.loc[len(statistic)] = ["Final Score",round(final_score,2)]

    return render(request,'EssayEvaluation/result.html',{'text':text,
                                                         "statistic": statistic.to_html(classes="styled-table", index=False),
                                                         'feedback_stat_criteria':feedback_stat_criteria,
                                                         'feedback_LLM':feedback_LLM,
                                                         'relevance_my':relevance(text),
                                                         'relevance_by_LLM':relevance_LLM,
                                                         })
    
