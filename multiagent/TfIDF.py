import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import fitz


def TfIDF(article):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(essays+[article])

    similarities=cosine_similarity(tfidf_matrix[-1:],tfidf_matrix[:-1]).flatten()

    average_sim = similarities.mean()
    print("Косинусное сходство первого эссе  всеми остальными:")
    for i, score in enumerate(similarities):
        print(f"Эссе {i + 1}: Сходство = {score:.2f}")

    print(average_sim)


def preprocess(text):
    return [word.lower() for word in text.split()]


def get_mean_vector(text,model):
    words = preprocess(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors,axis=0)



def extract_text_from_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    text=''

    for page_num in range(pdf_doc.page_count):
        page_text=pdf_doc.load_page(page_num)
        text+=page_text.get_text()

    return text

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Timon\Desktop\Trainee\essays.csv')
    essays = df['essay'].tolist()

    file_path = 'The Intelligence Age - Sam Altman.pdf'
    article_text = extract_text_from_pdf(file_path)
    # article_text = """It was a balancing act, one I didn’t always perform well. When my parents spoke to me in Turkish, I responded in English. At family gatherings, I stood at the edge of conversations, smiling."""

    
    
    tokenized_text = [preprocess(text) for text in essays]
    model = Word2Vec(tokenized_text,vector_size=100,window=5,min_count=1)

    article_vector = get_mean_vector(article_text,model)

    similarities = []
    for text in essays:
        essay_vector = get_mean_vector(text,model)
        similarity = cosine_similarity([article_vector],[essay_vector])[0][0]
        similarities.append(similarity)

    average_similarity = np.mean(similarities)
    print(f"Среднее сходство: {average_similarity:.2f}")

    if average_similarity < 0.2:
        print("Текст не является эссе.")
    else:
        print("Текст может быть эссе.")
