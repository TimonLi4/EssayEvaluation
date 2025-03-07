from pydantic_ai import Agent
from textblob import TextBlob
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import spacy
import math
import language_tool_python
import pyphen
import math
import string
import os
import fitz
import asyncio
import pandas as pd


from EssayEvaluation.multiagent.agent_model import agent


tokenizer_to_words = WordPunctTokenizer()
dic = pyphen.Pyphen(lang='en')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))


class TextProcessor:
    def __init__(self, text: str):
        self.text = text
        self.list_word, self.numWords, self.numSentence = self.tokenization()
        self.syllable_count = self.count_syllables()
        self.count_complex_sentences = self.count_complex_sentences()

    def tokenization(self):
        token_words = tokenizer_to_words.tokenize(self.text)
        list_word = [word for word in token_words if word not in string.punctuation]
        num_words = len(list_word)
        num_sentences = max(self.text.count("."), 1)  
        return list_word, num_words, num_sentences

    def count_syllables(self):
        return sum(len(dic.inserted(word).split("-")) for word in self.list_word)
    
    def count_complex_sentences(self):
        doc = nlp(self.text)
        return sum(1 for sent in doc.sents if any(token.dep_ == 'mark' for token in sent))


class BaseAgent(Agent):
    processor:TextProcessor
    def analyze(self):
        raise NotImplementedError("Каждый агент должен реализовать analyze().")

class AsyncAgentMixin:
    async def analyze(self):
        return await self._async_analyze()

class LexicalDiversityAgent(BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    def analyze(self):
        lex_diversity = len(set(self.processor.list_word)) / self.processor.numWords if self.processor.numWords else 0
        grade_lex_div = lex_diversity * 10
        return {"name": "Lexical diversity", "value": lex_diversity, "Grade": round(grade_lex_div, 2), "Weights": 0.15}

class DiffSent(BaseAgent):
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        diffsent = self.processor.numWords / self.processor.numSentence
        grade_diffsent = min(10, diffsent / 2)
        return {'name':'Complexity of sentences','value': diffsent, 'Grade': round(grade_diffsent, 2),'Weights':0.1}

class ReadabilityAgent(BaseAgent):
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        flesch_kincaid = 206.835 - 1.015 * (self.processor.numWords / self.processor.numSentence) - 84.6 * (self.processor.syllable_count  / self.processor.numWords)
        grade_flesch = 0.2 * flesch_kincaid + 4 if flesch_kincaid <= 30 else 10 if flesch_kincaid <= 50 else -0.15 * flesch_kincaid + 17.5
        return {'name':'Readability index','value': flesch_kincaid, 'Grade': round(grade_flesch, 2),'Weights':0.1}

class Tonality(BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    def analyze(self):
        polarity = TextBlob(self.processor.text).sentiment.polarity
        grade_pol = 2 * math.sqrt(10) * math.pow((math.sqrt(10) / 2), polarity)
        return {'name':'Emotional colouring','value': polarity, 'Grade': round(grade_pol, 2),'Weights':0.05}

class FSW(BaseAgent):
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        fsw = sum(1 for word in self.processor.list_word if word.lower() in stop_words) / self.processor.numWords
        grade_fsw = 10 * math.pow(0.064, fsw) if fsw < 0.4 else (20 / 3 * fsw + 4 / 3)
        return {'name':'Frequency stop word','value': fsw, 'Grade': round(grade_fsw, 2),'Weights':0.05}

class NumberOfComplexConstructions(BaseAgent):
    def __init__(self, processor):
        self.processor = processor

    def analyze(self):
        num_com = self.processor.count_complex_sentences / self.processor.numSentence
        grade_num_com = 4 * math.pow((10 / 4), num_com)
        return {'name':'Number of complex constructions','value': num_com, 'Grade': round(grade_num_com, 2),'Weights':0.1}

class Clarity(BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    def analyze(self):
        flesch_kincaid = 206.835 - 1.015 * (self.processor.numWords / self.processor.numSentence) - 84.6 * (self.processor.syllable_count  / self.processor.numWords)
        num_com = self.processor.count_complex_sentences / self.processor.numSentence
        clarity = flesch_kincaid - num_com
        grade_clarity = clarity / 10
        return {'name':'Clarity','value': clarity, 'Grade': round(grade_clarity, 2),'Weights':0.1}
    
class Grammar(BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    def analyze(self):
        tool = language_tool_python.LanguageToolPublicAPI('en')
        matches = tool.check(self.processor.text)

        num_errors = len(matches)
        error_ratio = num_errors/self.processor.numWords if self.processor.numWords>0 else 0
        score = max(10 - (error_ratio*50),0)

        return {'name':'Total Errors','value': num_errors, 'Grade': round(score, 2),'Weights':0.15}

class Punctuation(AsyncAgentMixin,BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    async def _async_analyze(self):
        response = await agent.run(f"Rate the punctuation quality of the following text on a scale from 0 to 10, where 0 means highly inaccurate and 10 means perfectly accurate. Print the number and additional comments: {self.processor.text}")
        return {'name':'Punctuation','value':'','Grade':response.data.split()[0],'Weights':0.05,'comments':response.data}

class StructureAgent(AsyncAgentMixin,BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    async def _async_analyze(self):
        response = await agent.run(f"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. Print the number and additional comments:{self.processor.text}")
        return {'name':'Punctuation','value':'','Grade':response.data.split()[0],'Weights':0.05,'comments':response.data}
    
class ContentAgent(AsyncAgentMixin,BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    async def _async_analyze(self):
        response = await agent.run(f"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. Print the number and additional comments:{self.processor.text}")
        return {'name':'Punctuation','value':'','Grade':response.data.split()[0],'Weights':0.05,'comments':response.data}
    
class CreativityAgent(AsyncAgentMixin,BaseAgent):
    def __init__(self, processor):
        self.processor = processor
    async def _async_analyze(self):
        response = await agent.run(f"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. Print the number and additional comments:{self.processor.text}")
        return {'name':'Punctuation','value':'','Grade':response.data.split()[0],'Weights':0.05,'comments':response.data}
    



class MASManager:
    """Менеджер мультиагентной системы."""
    def __init__(self, text: str):
        self.processor = TextProcessor(text)
        self.agents = [
            LexicalDiversityAgent(self.processor),
            DiffSent(self.processor),
            ReadabilityAgent(self.processor),
            Tonality(self.processor),            
            FSW(self.processor),            
            NumberOfComplexConstructions(self.processor),            
            Clarity(self.processor),                     
            Grammar(self.processor),
            Punctuation(self.processor),
            StructureAgent(self.processor),                     
            ContentAgent(self.processor),
            CreativityAgent(self.processor)
        ]

    async def run(self):
        sync_results = []
        async_results = []
        
        # Разделяем синхронные и асинхронные агенты
        for ag in self.agents:
            if isinstance(ag, AsyncAgentMixin):
                async_results.append(ag.analyze())
            else:
                sync_results.append(ag.analyze())
        
        # Параллельно выполняем асинхронные задачи
        async_part = await asyncio.gather(*async_results)
        return sync_results + async_part
    

def extract_text_from_pdf(file_path):
    pdf_doc = fitz.open(file_path)
    text=''

    for page_num in range(pdf_doc.page_count):
        page_text=pdf_doc.load_page(page_num)
        text+=page_text.get_text()

    return text




async def main():
    text = extract_text_from_pdf('essay.pdf')
    print(text)
    manager = MASManager(text)
    results = await manager.run()
    print(pd.DataFrame(results))

if __name__ =='__main__':
    asyncio.run(main())    
