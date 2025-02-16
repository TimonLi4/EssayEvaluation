from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

import os
from dotenv import load_dotenv
import pandas as pd
import fitz

load_dotenv()

file_essay = os.getenv('file_essay')
file_path = os.getenv('FILE_PATH')
data_path = os.getenv('DATA_PATH')
file_article = os.getenv('FILE_ARTICLE')


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def extract_text_from_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    text=''

    for page_num in range(pdf_doc.page_count):
        page_text=pdf_doc.load_page(page_num)
        text+=page_text.get_text()

    return text


def Relevance(new_text_topics):
    print([prob for _,prob in new_text_topics])
    # print(sum(prob for _, prob in new_text_topics))
    # print('\n\n')
    average_probability = sum(prob for _, prob in new_text_topics) / len(new_text_topics)
    max_probability = max(prob for _, prob in new_text_topics)
    min_probability = min(prob for _, prob in new_text_topics)

    answer = ''

    print((min_probability+max_probability)/2 - average_probability, min_probability)

    if ((min_probability+max_probability)/2 - average_probability) > min_probability:
        print("article")
        answer = 'article'
    else:
        print("essay")
        answer = 'essay'

    return f"{(min_probability+max_probability)/2 - average_probability, min_probability}", answer


def Create_LDA_essay(file_essay):
    if os.path.exists("lda_model_essay\lda_model.model") and os.path.exists("lda_model_essay\dictionary.dict"):
        ldamodel = LdaModel.load("lda_model_essay\lda_model.model")
        dictionary = Dictionary.load("lda_model_essay\dictionary.dict")
    else:
        
        df = pd.read_csv(file_essay)
        doc_clean = [clean(doc).split() for doc in df['essay']]
        
        dictionary = Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        
        ldamodel = LdaModel(doc_term_matrix, num_topics=6, id2word=dictionary, passes=50)
        
        ldamodel.save("lda_model_essay\lda_model.model")
        dictionary.save("lda_model_essay\dictionary.dict")
        
    
    return dictionary, ldamodel

def Create_LDA_article(file_article):
    model_path = os.path.join("lda_model_article", "lda_model.model")
    dict_path = os.path.join("lda_model_article", "dictionary.dict")

    if os.path.exists(model_path) and os.path.exists(dict_path):
        ldamodel = LdaModel.load(model_path)
        dictionary = Dictionary.load(dict_path)
    else:
        
        os.makedirs("lda_model_article", exist_ok=True)

        df = pd.read_csv(file_article)
        doc_clean = [clean(doc).split() for doc in df['text']]

        dictionary = Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

        ldamodel = LdaModel(doc_term_matrix, num_topics=8, id2word=dictionary, passes=50)

        ldamodel.save(model_path)
        dictionary.save(dict_path)
    
    return  dictionary,ldamodel


# def classify_text(text, lda_essay, dictionary_essay, lda_article, dictionary_article):
#     # Очистка текста
#     text_clean = clean(text).split()

#     # Преобразование текста в BoW
#     bow_essay = dictionary_essay.doc2bow(text_clean)
#     bow_article = dictionary_article.doc2bow(text_clean)

#     # Получаем распределение тем
#     topics_essay = lda_essay.get_document_topics(bow_essay)
#     topics_article = lda_article.get_document_topics(bow_article)

#     # Суммируем вероятности тем
#     score_essay = [prob for _, prob in topics_essay]
#     score_article = [prob for _, prob in topics_article]

#     print(f"Score Essay: {score_essay} ->  {sum(score_essay)/len(topics_essay)} \nScore Article: {score_article} -> {sum(score_article)/len(topics_article)}")

#     # Классификация
#     # if score_essay > score_article:
#     #     return "Эссе"
#     # else:
#     #     return "Статья"


if __name__ == '__main__':

    dictionary_essay, ldamodel_essay = Create_LDA_essay(file_essay)
    dictionary_article, ldamodel_article = Create_LDA_article(file_article)


    # Обработка нового текста
    # data_path = r'C:\Users\Timon4\Desktop\projectTrainee\other\ARTICLE\bbc_news_text_complexity_summarization.csv'
    new_text = pd.read_csv(data_path)['essay'][1] # file_essay data_path
    
    # new_text = """"""
    new_text = extract_text_from_pdf(file_path)
    print(new_text)


    new_text_clean = clean(new_text)
    new_text_tokens = new_text_clean.split() 

    
    new_bow = dictionary_essay.doc2bow(new_text_tokens)

    new_text_topics = ldamodel_essay.get_document_topics(new_bow)
    
    Relevance(new_text_topics)


    # classify_text(new_text, ldamodel_essay, dictionary_essay, ldamodel_article, dictionary_article)
