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
    average_probability = sum(prob for _, prob in new_text_topics) / len(new_text_topics)
    max_probability = max(prob for _, prob in new_text_topics)

    # threshold = (average_probability + max_probability) / 2
    # print(f"Пороговое значение: {threshold:.2f}")

    # sum_new_text = sum(prob for _, prob in new_text_topics)
    # print(sum_new_text)

    print((average_probability+max_probability)/2, max_probability)

    if (average_probability+max_probability)/2 >= max_probability:
        print("Текст соответствует основным темам (это, вероятно, эссе).")
    else:
        print("Текст не соответствует основным темам (это, возможно, статья).")

    print('\n',new_text_topics)


def Create_LDA(file_essay):
    if os.path.exists("lda_model.model") and os.path.exists("dictionary.dict"):
        ldamodel = LdaModel.load("lda_model.model")
        dictionary = Dictionary.load("dictionary.dict")
    else:
        
        df = pd.read_csv(file_essay)
        doc_clean = [clean(doc).split() for doc in df['essay']]
        
        dictionary = Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        
        ldamodel = LdaModel(doc_term_matrix, num_topics=6, id2word=dictionary, passes=50)
        
        ldamodel.save("lda_model.model")
        dictionary.save("dictionary.dict")
        
    
    return dictionary, ldamodel


if __name__ == '__main__':

    dictionary, ldamodel = Create_LDA(file_essay)

    # Обработка нового текста
    
    new_text = pd.read_csv(data_path)['essay'][0] # file_essay data_path
    # new_text = 
    new_text = extract_text_from_pdf(file_path)
    print(new_text)
    new_text_clean = clean(new_text)
    new_text_tokens = new_text_clean.split()  # Преобразуем строку в список токен

    # Преобразуем новый текст в bag-of-words
    new_bow = dictionary.doc2bow(new_text_tokens)

    # Получаем распределение тем для нового текста
    new_text_topics = ldamodel.get_document_topics(new_bow)
    
    Relevance(new_text_topics)
    # threshold = 0.2  # Установите порог вероятности для релевантности
    # print("\nРаспределение тем для нового текста:")
    # is_relevant = False
    # for topic_id, prob in new_text_topics:
    #     print(f"Тема {topic_id + 1} с вероятностью {prob:.2f}")
    #     if prob > threshold:
    #         is_relevant = True
    
    # if is_relevant:
    #     print("\nНовый текст релевантен основным темам.")
    # else:
    #     print("\nНовый текст нерелевантен основным темам.")