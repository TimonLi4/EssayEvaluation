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

    return f"{(average_probability+max_probability)/2} --- {max_probability}"


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
    new_text = """In recent years, online learning has become an essential part of the modern education system. With the rapid development of digital technologies, this form of education offers numerous advantages for both school students and university learners. In this article, we will explore the key benefits of online learning and how it can improve the educational experience.

1. Flexibility and Accessibility
One of the primary advantages of online learning is its flexibility. Students can choose the most convenient time for their studies and adjust their schedule to meet personal needs. This is especially valuable for those who balance education with work or family commitments.

Moreover, online learning provides access to education for individuals living in remote areas with limited educational opportunities. With just an internet connection, anyone can enroll in quality courses offered by institutions worldwide.

2. Personalized Learning Experience
Online platforms enable students to tailor their learning process according to their individual preferences. Learners can set their own pace, revisit complex topics, or skip over material they’ve already mastered. This personalized approach ensures a deeper understanding of the subject.

3. Cost and Time Efficiency
Traditional education often involves additional expenses for transportation, accommodation, and meals. Online learning significantly reduces these costs. Students can also save time on commuting, allowing them to focus on additional projects or personal development.

4. A Wide Range of Courses
Today, there is an abundance of online courses covering virtually every field — from programming and design to psychology and languages. This variety enables students to explore new interests and acquire skills that are in demand in the modern job market.

Conclusion
Online learning is a transformative step forward in education, offering students greater flexibility and new opportunities. However, success in this format requires self-motivation and discipline. While online learning is a powerful tool, it should complement traditional education rather than completely replace it."""
    # new_text = extract_text_from_pdf(file_path)
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