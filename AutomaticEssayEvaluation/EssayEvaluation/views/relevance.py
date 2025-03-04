from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re

import os

import pandas as pd

from django.conf import settings



# file_essay = os.getenv('file_essay')
# file_path = os.getenv('FILE_PATH')
# data_path = os.getenv('DATA_PATH')
# file_article = os.getenv('FILE_ARTICLE')


stop_words = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    doc = doc.lower()  # Lowercase
    doc = re.sub(r'\d+', '', doc)  # Deleting numbers
    doc = " ".join(word for word in doc.split() if word not in stop_words)  # Removing stop words
    doc = ''.join(ch for ch in doc if ch not in exclude)  # Removing punctuation
    doc = " ".join(lemma.lemmatize(word, pos='v') for word in doc.split())  # Lemmatization (including verbs)
    doc = re.sub(r'\s+', ' ', doc).strip()  # Removing unnecessary spaces
    return doc



def Check_stat(new_text_topics):
    average_probability = sum(prob for _, prob in new_text_topics) / len(new_text_topics)
    max_probability = max(prob for _, prob in new_text_topics)
    min_probability = min(prob for _, prob in new_text_topics)
    
    value = (min_probability+max_probability)/2 - average_probability

    return value

def Create_LDA_essay(file_essay):
    model_path = os.path.join(settings.MEDIA_ROOT, "lda_model_essay", "lda_model.model")
    dict_path = os.path.join(settings.MEDIA_ROOT, "lda_model_essay", "dictionary.dict")

    if os.path.exists(model_path) and os.path.exists(dict_path):
        ldamodel = LdaModel.load(model_path)
        dictionary = Dictionary.load(dict_path)
    else:
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "lda_model_essay"), exist_ok=True)
        df = pd.read_csv(file_essay)
        doc_clean = [clean(doc).split() for doc in df['essay']]
        
        dictionary = Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        
        ldamodel = LdaModel(doc_term_matrix, num_topics=6, id2word=dictionary, passes=50)
        
        ldamodel.save(model_path)
        dictionary.save(dict_path)
        
    return dictionary, ldamodel

def Create_LDA_article(file_article):
    model_path = os.path.join(settings.MEDIA_ROOT, "lda_model_article", "lda_model.model")
    dict_path = os.path.join(settings.MEDIA_ROOT, "lda_model_article", "dictionary.dict")

    if os.path.exists(model_path) and os.path.exists(dict_path):
        ldamodel = LdaModel.load(model_path)
        dictionary = Dictionary.load(dict_path)
    else:
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "lda_model_article"), exist_ok=True)

        df = pd.read_csv(file_article)
        doc_clean = [clean(doc).split() for doc in df['text']]

        dictionary = Dictionary(doc_clean)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

        ldamodel = LdaModel(doc_term_matrix, num_topics=8, id2word=dictionary, passes=50)

        ldamodel.save(model_path)
        dictionary.save(dict_path)
    
    return  dictionary,ldamodel

def Score_all(new_text,file_essay,file_article):

    dictionary_article, ldamodel_article = Create_LDA_article(file_article)
    dictionary_essay, ldamodel_essay = Create_LDA_essay(file_essay)

    new_text_clean = clean(new_text)
    new_text_tokens = new_text_clean.split() 
    
    new_bow = dictionary_essay.doc2bow(new_text_tokens)

    new_text_topics_essay = ldamodel_essay.get_document_topics(new_bow)

    new_bow = dictionary_article.doc2bow(new_text_tokens)

    new_text_topics_article = ldamodel_article.get_document_topics(new_bow)
    
    
    value1 = Check_stat(new_text_topics_essay)
    
    value2 = Check_stat(new_text_topics_article)

    return value1,value2
