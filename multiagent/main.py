from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from textblob import TextBlob
from nltk.tokenize import word_tokenize,sent_tokenize, WordPunctTokenizer
from nltk.corpus import stopwords
import spacy
import pandas as pd
import string
import pyphen
import os

from dotenv import load_dotenv





load_dotenv()

key_api = os.getenv('API_KEY')


tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
tokenizer_to_words = WordPunctTokenizer()
dic = pyphen.Pyphen(lang='en')
nlp = spacy.load('en_core_web_sm')


model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")


stop_words = set(stopwords.words('english'))


df = pd.read_csv(r'C:\Users\Timon\Desktop\Trainee\Automatic-Essay-Scoring-master\Processed_data.csv')


def CountSyllabels(words):
    summSyllabels = 0

    for word in words:
        hyphenated_word = dic.inserted(word)
        summSyllabels+=len(hyphenated_word.split('-'))

    return summSyllabels


def CountComplexSentences(text):
    doc = nlp(text)

    complexCount = 0
    for sent in doc.sents:
        if any(token.dep_ == 'mark' for token in sent):
            complexCount+=1
    return complexCount


def Tokenization(input_text):
    token_words = tokenizer_to_words.tokenize(input_text)
    list_word = [word for word in token_words if word not in string.punctuation]
    dotCount = 0
    for i in token_words:
        if i == '.':
            dotCount += 1

    if dotCount == 0:
        dotCount = 1
    
    return list_word, len(list_word),dotCount


def DiffSent(numWords,numSentence):
    return numWords/numSentence


def Lex_mix(text):
    return len(set(text))/len(text)


def FleschKincaid(difSent,wordCount,countSyllabels):
    return 206.835-1.015*difSent - 84.6*countSyllabels/wordCount


def Tonality(input_text):
    blob = TextBlob(input_text)
    return blob


def FSW(list_word,wordCount):
    return sum(1 for word in list_word if word.lower() in stop_words)/wordCount

# print(Tonality(input_text).sentiment)

def NumberOfComplexConstructions(difSent,numOfSent):
    return difSent/numOfSent


def Clarity(flesgKincaid):
    pass



if __name__ =='__main__':
    # print(df.head())
    input_text = df.iloc[2]['essay']

    print(input_text)
    # input_text = "This product is great, but the delivery was terrible.This product is great, but the delivery was terrible."

    list_word, numWords,numSentence = Tokenization(input_text)


    print('Lexical diversity - ',Lex_mix(list_word))


    difSent =DiffSent(numWords,numSentence) 
    print('Complexity of sentences - ',difSent)


    flesgKincaid = FleschKincaid(difSent,numWords,CountSyllabels(list_word))
    print('Readability index - ',flesgKincaid)


    print('Formality of style - ',)


    print('Emotional colouring - ',Tonality(input_text).sentiment)


    print('Frequency stop word - ',FSW(list_word,numWords))

    countComplexSent = CountComplexSentences(input_text)
    numOfComplexConstructions = NumberOfComplexConstructions(countComplexSent,numSentence)
    print('Number of complex constructions - ',numOfComplexConstructions)


    print('Clarity of text - ',flesgKincaid-numOfComplexConstructions)


    print(key_api)


