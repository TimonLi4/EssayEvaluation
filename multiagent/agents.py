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




load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')

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
        return {'Lexical diversity': lex_diversity, 'Grade': grade_lex_div,'Weights':0.15}

class DiffSent(TextAnalysis):
    def analyze(self):
        diffsent = self.numWords / self.numSentence
        grade_diffsent = min(10, diffsent / 2)
        return {'Complexity of sentences': diffsent, 'Grade': round(grade_diffsent, 2),'Weights':0.1}

class FleschKincaid(TextAnalysis):
    def count_syllables(self):
        return sum(len(dic.inserted(word).split('-')) for word in self.list_word)

    def analyze(self):
        flesch_kincaid = 206.835 - 1.015 * (self.numWords / self.numSentence) - 84.6 * (self.count_syllables() / self.numWords)
        grade_flesch = 0.2 * flesch_kincaid + 4 if flesch_kincaid <= 30 else 10 if flesch_kincaid <= 50 else -0.15 * flesch_kincaid + 17.5
        return {'Readability index': flesch_kincaid, 'Grade': round(grade_flesch, 2),'Weights':0.1}

class Tonality(TextAnalysis):
    def analyze(self):
        blob = TextBlob(self.text)
        polarity = blob.sentiment.polarity
        grade_pol = 2 * math.sqrt(10) * math.pow((math.sqrt(10) / 2), polarity)
        return {'Emotional colouring': polarity, 'Grade': round(grade_pol, 2),'Weights':0.05}

class FSW(TextAnalysis):
    def analyze(self):
        fsw = sum(1 for word in self.list_word if word.lower() in stop_words) / self.numWords
        grade_fsw = 10 * math.pow(0.064, fsw) if fsw < 0.4 else (20 / 3 * fsw + 4 / 3)
        return {'Frequency stop word': fsw, 'Grade': round(grade_fsw, 2),'Weights':0.05}

class NumberOfComplexConstructions(TextAnalysis):
    def count_complex_sentences(self):
        doc = nlp(self.text)
        return sum(1 for sent in doc.sents if any(token.dep_ == 'mark' for token in sent))

    def analyze(self):
        num_com = self.count_complex_sentences() / self.numSentence
        grade_num_com = 4 * math.pow((10 / 4), num_com)
        return {'Number of complex constructions': num_com, 'Grade': round(grade_num_com, 2),'Weights':0.1}

class Clarity(TextAnalysis):
    def __init__(self, text):
        super().__init__(text)
        self.flesch_kincaid = FleschKincaid(text).analyze()['Readability index']
        self.num_com = NumberOfComplexConstructions(text).analyze()['Number of complex constructions']

    def analyze(self):
        clarity = self.flesch_kincaid - self.num_com
        grade_clarity = clarity / 10
        return {'Clarity': clarity, 'Grade': round(grade_clarity, 2),'Weights':0.1}

class Grammar(TextAnalysis):
    def analyze(self):
        tool = language_tool_python.LanguageToolPublicAPI('en')
        matches = tool.check(self.text)

        num_errors = len(matches)
        error_ratio = num_errors/self.numWords if self.numWords>0 else 0
        score = max(10 - (error_ratio*50),0)

        return {'Total Errors': num_errors, 'Grade': round(score, 2),'Weights':0.15}

class Punctuation(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Evaluate the punctuation in the given text on a scale from 0 to 10.Output only the number, without any explanations, words, or symbols. Text: {self.text}"}])
        return {'Punctuation':'','Grade':response["message"]["content"].split()[0],'Weights':0.05}

class StructureAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. Output only the number without any additional comments:{self.text}"}])
        return {'Structure':'','Grade':response["message"]["content"].split()[0],'Weights':0.05}

class ContentAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. Output only the number without any additional comments:{self.text}"}])
        return {'Information':'','Grade': response["message"]["content"].split()[0],'Weights':0.05}


class CreativityAgent(TextAnalysis):
    def analyze(self):
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. Print only the number without any additional comments:{self.text}"}])
        return {'Creativity':'','Grade': response["message"]["content"].split()[0],'Weights':0.05}


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



if __name__ =='__main__':
    df = pd.read_csv(DATA_PATH)
    text = df.iloc[0]['essay']

    final_score = 0
    mas = MAS(text)
    result = mas.evaluate()  
    for i in result:
        print(i)
        final_score+=float(i['Grade'])*i['Weights']
        
    print('Final score: ',round(final_score,2))


