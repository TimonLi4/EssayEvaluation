import pandas as pd
import os
from grammar import AsessmentCreativety,AsessmentSturcture,CheckGrammar,Information
from dotenv import load_dotenv
import tiktoken

from TechnicalAspects import Tokenization,Lex_mix, DiffSent,FleschKincaid,CountComplexSentences,CountSyllabels,Tonality,NumberOfComplexConstructions,FSW,Clarity


load_dotenv()


DATA_PATH = os.getenv('DATA_PATH')
encoding = tiktoken.encoding_for_model('gpt-4')


# key_api = os.getenv('API_KEY')


if __name__ =='__main__':
    df = pd.read_csv(DATA_PATH)
    text = df.iloc[5]['essay']

    print(text)

    list_word, numWords,numSentence = Tokenization(text)

    lex_mix, grade_lex_mix = Lex_mix(list_word)
    print('Lexical diversity - ',lex_mix, f'Grade - {grade_lex_mix}')

    difSent,grade_difsent =DiffSent(numWords,numSentence) 
    print('Complexity of sentences - ',difSent,f'Grade - {grade_difsent}')

    flesgKincaid,grade_flesgKincaid = FleschKincaid(difSent,numWords,CountSyllabels(list_word))
    print('Readability index - ',flesgKincaid, 'Grade - ',grade_flesgKincaid)

    polarity,grade_polarity = Tonality(text)
    print('Emotional colouring - ',polarity,'Grade - ',grade_polarity )

    fsw,grade_fsw = FSW(list_word,numWords)
    print('Frequency stop word - ',fsw,'Grade - ',grade_fsw)

    countComplexSent = CountComplexSentences(text)
    numOfComplexConstructions,grade_numOfComplexConstructions = NumberOfComplexConstructions(countComplexSent,numSentence)
    print('Number of complex constructions - ',numOfComplexConstructions,'Grade - ',grade_numOfComplexConstructions)

    clarity,grade_clarity = Clarity(flesgKincaid, countComplexSent)
    print('Clarity of text - ', clarity,'Grade - ',grade_clarity)

    print("---------------------------------------------------------------------")

    grammar = CheckGrammar(text)
    grade_grammar = float(grammar.split()[0])

    creativety = AsessmentCreativety(text)
    grade_creativety = float(creativety.split()[0])

    structure = AsessmentSturcture(text)
    grade_structure = float(structure.split()[0])

    information = Information(text)
    grade_information = float(information.split()[0])

    print("---------------------------------------------------------------------")
    
    print('Assessment of grammar and punctuation - ', grammar)
    
    print("--------------------------")
    
    print('Creativity assessment - ',creativety)
    
    print("--------------------------")
    
    print('Structure assessment - ',structure)
    
    print("--------------------------")
    
    print('Informative assessment - ',information)
    
    print("---------------------------------------------------------------------")
    
    final_grade = 0.2*grade_clarity+0.2*grade_information+0.15*grade_structure+0.15*grade_grammar+0.1*grade_difsent+0.07*grade_lex_mix+0.05*grade_polarity+0.05*grade_creativety+0.03*grade_flesgKincaid+0.05*grade_numOfComplexConstructions+0.05*grade_fsw
    print('final grade - ',final_grade)


    print("---------------------------------------------------------------------")
    
    print('Token Grammar = ', len(encoding.encode(text+"Rate the grammar and punctuation of the following text on a scale from 0 to 10, where 0 means full of errors and 10 means perfect. Output only the number without any additional comments:")))
    
    print('Token AsessmentCreativety = ',len(encoding.encode(text+"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. Print only the number without any additional comments:")))
    
    print('Token AsessmentSturcture = ',len(encoding.encode(text+"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. Output only the number without any additional comments:")))
    
    print('Token informativeness = ',len(encoding.encode(text+"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. Output only the number without any additional comments:")))
