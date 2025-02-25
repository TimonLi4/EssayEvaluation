from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import spacy
import math
import string
import pandas as pd
import os
from dotenv import load_dotenv
import language_tool_python
import ollama

import pyphen

from together import Together

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
API_KEY = os.getenv('API_KEY')
client = Together(api_key=API_KEY)

tokenizer_to_words = WordPunctTokenizer()
dic = pyphen.Pyphen(lang='en')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

class TextAnalysis:
    def __init__(self, text):
        self.text = text
        self.list_word, self.numWords, self.numSentence = self.tokenization(text)
        

    def tokenization(self, input_text):
        token_words = tokenizer_to_words.tokenize(input_text)
        list_word = [word for word in token_words if word not in string.punctuation]
        dotCount = token_words.count('.')
        dotCount = max(dotCount, 1)  # Избегаем деления на 0
        return list_word, len(list_word), dotCount

class LexMin(TextAnalysis):
    def analyze(self):
        lex_diversity = len(set(self.list_word)) / len(self.list_word) if self.list_word else 0
        grade_lex_div = lex_diversity * 10
        return {'name':'Lexical diversity','value': lex_diversity, 'Grade': round(grade_lex_div,2),'Weights':0.15}

class DiffSent(TextAnalysis):
    def analyze(self):
        diffsent = self.numWords / self.numSentence
        grade_diffsent = min(10, diffsent / 2)
        return {'name':'Complexity of sentences','value': diffsent, 'Grade': round(grade_diffsent, 2),'Weights':0.1}

class FleschKincaid(TextAnalysis):
    def count_syllables(self):
        return sum(len(dic.inserted(word).split('-')) for word in self.list_word)

    def analyze(self):
        flesch_kincaid = 206.835 - 1.015 * (self.numWords / self.numSentence) - 84.6 * (self.count_syllables() / self.numWords)
        grade_flesch = 0.2 * flesch_kincaid + 4 if flesch_kincaid <= 30 else 10 if flesch_kincaid <= 50 else -0.15 * flesch_kincaid + 17.5
        return {'name':'Readability index','value': flesch_kincaid, 'Grade': round(grade_flesch, 2),'Weights':0.1}

class Tonality(TextAnalysis):
    def analyze(self):
        blob = TextBlob(self.text)
        polarity = blob.sentiment.polarity
        grade_pol = 2 * math.sqrt(10) * math.pow((math.sqrt(10) / 2), polarity)
        return {'name':'Emotional colouring','value': polarity, 'Grade': round(grade_pol, 2),'Weights':0.05}

class FSW(TextAnalysis):
    def analyze(self):
        fsw = sum(1 for word in self.list_word if word.lower() in stop_words) / self.numWords
        grade_fsw = 10 * math.pow(0.064, fsw) if fsw < 0.4 else (20 / 3 * fsw + 4 / 3)
        return {'name':'Frequency stop word','value': fsw, 'Grade': round(grade_fsw, 2),'Weights':0.05}

class NumberOfComplexConstructions(TextAnalysis):
    def count_complex_sentences(self):
        doc = nlp(self.text)
        return sum(1 for sent in doc.sents if any(token.dep_ == 'mark' for token in sent))

    def analyze(self):
        num_com = self.count_complex_sentences() / self.numSentence
        grade_num_com = 4 * math.pow((10 / 4), num_com)
        return {'name':'Number of complex constructions','value': num_com, 'Grade': round(grade_num_com, 2),'Weights':0.1}

class Clarity(TextAnalysis):
    def __init__(self, text):
        super().__init__(text)
        self.flesch_kincaid = FleschKincaid(text).analyze()['value']
        self.num_com = NumberOfComplexConstructions(text).analyze()['value']

    def analyze(self):
        clarity = self.flesch_kincaid - self.num_com
        grade_clarity = clarity / 10
        return {'name':'Clarity','value': clarity, 'Grade': round(grade_clarity, 2),'Weights':0.1}

class Grammar(TextAnalysis):
    def analyze(self):
        tool = language_tool_python.LanguageToolPublicAPI('en')
        matches = tool.check(self.text)

        num_errors = len(matches)
        error_ratio = num_errors/self.numWords if self.numWords>0 else 0
        score = max(10 - (error_ratio*50),0)

        return {'name':'Total Errors','value': num_errors, 'Grade': round(score, 2),'Weights':0.15}
    

# class Punctuation(TextAnalysis):
#     def analyze(self):

#         response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Evaluate the punctuation in the given text on a scale from 0 to 10. First, print the number, followed by additional comments. Text: {self.text}"}],)

#         # response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Evaluate the punctuation in the given text on a scale from 0 to 10.Print the number and additional comments: {self.text}"}])
#         return {'name':'Punctuation','value':'','Grade':response.choices[0].message.content.split()[0],'Weights':0.05,'comments':response.choices[0].message.content}

# class StructureAgent(TextAnalysis):
#     def analyze(self):
#         response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. First, print the number, followed by additional comments. text:{self.text}"}])
#         return {'name':'Structure','value':'','Grade':response.choices[0].message.content.split()[0],'Weights':0.05,'comments':response.choices[0].message.content}

# class ContentAgent(TextAnalysis):
#     def analyze(self):
#         response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. First, print the number, followed by additional comments. Text:{self.text}"}])
#         return {'name':'Information','value':'','Grade': response.choices[0].message.content.split()[0],'Weights':0.05,'comments':response.choices[0].message.content}


# class CreativityAgent(TextAnalysis):
#     def analyze(self):
#         response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. First, print the number, followed by additional comments. Text:{self.text}"}])
#         return {'name':'Creativity','value':'','Grade': response.choices[0].message.content.split()[0],'Weights':0.05, 'comments':response.choices[0].message.content}

# def Relevance(text):
#     response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Analyze the following text and determine whether it is an essay or an article. An essay typically contains personal reflections, a subjective point of view, and an individual writing style, whereas an article aims for objectivity, structure, and the use of facts. Answer with one word: 'essay' or 'article'.  Text:{text}"}])
#     return response.choices[0].message.content

###################################################################
class Punctuation(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Evaluate the punctuation in the given text on a scale from 0 to 10.Print the number and additional comments: {self.text}"}])
        return {'name':'Punctuation','value':'','Grade':response["message"]["content"].split()[0],'Weights':0.05,'comments':response["message"]["content"]}

class StructureAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. Print the number and additional comments:{self.text}"}])
        return {'name':'Structure','value':'','Grade':response["message"]["content"].split()[0],'Weights':0.05,'comments':response["message"]["content"]}

class ContentAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. Print the number and additional comments:{self.text}"}])
        return {'name':'Information','value':'','Grade': response["message"]["content"].split()[0],'Weights':0.05,'comments':response["message"]["content"]}


class CreativityAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. Print the number and additional comments:{self.text}"}])
        return {'name':'Creativity','value':'','Grade': response["message"]["content"].split()[0],'Weights':0.05, 'comments':response["message"]["content"]}

def Relevance(text):
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Analyze the following text and determine whether it is an essay or an article. An essay typically contains personal reflections, a subjective point of view, and an individual writing style, whereas an article aims for objectivity, structure, and the use of facts. Answer with one word: 'essay' or 'article'. Text: {text}"}])
    return response["message"]["content"]

###################################################################

class MAS:
    def __init__(self, text):
        self.agents = [
            LexMin(text),
            DiffSent(text),
            FleschKincaid(text),
            Tonality(text),
            FSW(text),
            NumberOfComplexConstructions(text),
            Clarity(text),
            Grammar(text),
            Punctuation(text),
            CreativityAgent(text),
            ContentAgent(text),
            StructureAgent(text),
        ]

    def evaluate(self):
        results = []
        for agent in self.agents:
            results.append(agent.analyze())
        return results



def Feedback_fromLLM(df):
    text = ''
    for i in range(df.shape[0]):
        
        if not pd.isna(df.iloc[i]['comments']): 
            text+= f"{df.iloc[i]['name']} | {df.iloc[i]['comments']} \n\n\n"
    
    return text


def FeedBack_stat_criteria(df):
    text = ''
    for i in range(df.shape[0]):
        if pd.isna(df.iloc[i]['comments']):      
            text += f"{df.iloc[i]['name']} - {df.iloc[i]['value']} | Grade - {df.iloc[i]['Grade']} \n"

    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Based on the provided text metrics and grades, generate detailed feedback on the text's quality. Analyze each metric and give an evaluation of the text's strengths and areas for improvement. Focus on lexical diversity, sentence complexity, readability, emotional coloring, and clarity. Conclude with overall feedback on the text's readability, target audience suitability, and possible improvements. Metrics: {text}"}])
    return response["message"]["content"]


############################################
# def FeedBack_stat_criteria(df):

    # text = ''
    #     for i in range(df.shape[0]):
    #         if pd.isna(df.iloc[i]['comments']):      
    #             text += f"{df.iloc[i]['name']} - {df.iloc[i]['value']} | Grade - {df.iloc[i]['Grade']} \n"
    # response = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": f"Based on the provided text metrics and grades, generate detailed feedback on the text's quality. Analyze each metric and give an evaluation of the text's strengths and areas for improvement. Focus on lexical diversity, sentence complexity, readability, emotional coloring, and clarity. Conclude with overall feedback on the text's readability, target audience suitability, and possible improvements. Metrics: {text}"}])
    # return response.choices[0].message.content
