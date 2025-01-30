from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import spacy
import math
import string
import pyphen


tokenizer_to_words = WordPunctTokenizer()
dic = pyphen.Pyphen(lang='en')
nlp = spacy.load('en_core_web_sm')



stop_words = set(stopwords.words('english'))


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
    diffsent = numWords/numSentence
    grade = 0
    if diffsent >=1 and diffsent <10:
        grade = 4
        return diffsent, round(grade,2)
    elif diffsent >=20:
        grade = 10
        return diffsent, round(grade,2)
    elif diffsent<20 and diffsent>=10:
        grade = diffsent/2
        return diffsent, round(grade,2)
    


def Lex_mix(text):
    lex_mix = len(set(text))/len(text)
    grade = 10 * lex_mix
    return lex_mix, round(grade,2)


def FleschKincaid(difSent,wordCount,countSyllabels):
    flesgKincaid = 206.835-1.015*difSent - 84.6*countSyllabels/wordCount
    grade = 0
    if flesgKincaid >=0 and flesgKincaid<= 30:
        grade = 0.2*flesgKincaid+4
        return flesgKincaid, round(grade,2)
    elif flesgKincaid>30 and flesgKincaid<=50:
        return flesgKincaid, 10
    elif flesgKincaid>50 and flesgKincaid<=90:
        grade = -0.15*flesgKincaid+17.5
        return flesgKincaid, round(grade,2)
    elif flesgKincaid>90 and flesgKincaid<=100:
        return flesgKincaid, 4


def Tonality(input_text):
    blob = TextBlob(input_text)
    polarity = blob.sentiment.polarity
    grade = 2*math.sqrt(10)*math.pow((math.sqrt(10)/2),polarity)
    return polarity, round(grade,2) 


def FSW(list_word,wordCount):
    fsw = sum(1 for word in list_word if word.lower() in stop_words)/wordCount
    grade = 0
    if fsw>=0 and fsw<0.4:
        grade = 10*math.pow(0.064,fsw)
        return fsw,grade
    elif fsw>=0.4 and fsw<=1:
        grade = 20/3*fsw+4/3
        return fsw,round(grade,2) 
    


def NumberOfComplexConstructions(difSent,numOfSent):
    numCom = difSent/numOfSent 
    grade = 4*math.pow((10/4),numCom)
    return numCom,round(grade,2)

def Clarity(flesgKincaid,countComplexSent):
    clarity = flesgKincaid-countComplexSent
    grade = clarity/10
    return clarity,round(grade,2)
