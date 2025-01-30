import pandas as pd
# import os
from grammar import AsessmentCreativety,AsessmentSturcture,CheckGrammar,Information
from dotenv import load_dotenv
import tiktoken

from TechnicalAspects import Tokenization,Lex_mix, DiffSent,FleschKincaid,CountComplexSentences,CountSyllabels,Tonality,NumberOfComplexConstructions,FSW,Clarity


load_dotenv()

encoding = tiktoken.encoding_for_model('gpt-4')


# key_api = os.getenv('API_KEY')



if __name__ =='__main__':
    df = pd.read_csv(r'C:\Users\Timon4\Desktop\projectTrainee\other\Automatic-Essay-Scoring-master\Processed_data.csv')
    text = df.iloc[2]['essay']

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
    grade_grammar = float(grammar)

    creativety = AsessmentCreativety(text)
    grade_creativety = float(creativety)

    structure = AsessmentSturcture(text)
    grade_structure = float(structure)

    information = Information(text)
    grade_information = float(information)

    print("---------------------------------------------------------------------")
    print(grammar)
    print("--------------------------")
    print(creativety)
    print("--------------------------")
    print(structure)
    print("--------------------------")
    print(information)
    print("---------------------------------------------------------------------")
    final_grade = 0.2*grade_clarity+0.2*grade_information+0.15*grade_structure+0.15*grade_grammar+0.1*grade_difsent+0.07*grade_lex_mix+0.05*grade_polarity+0.05*grade_creativety+0.03*grade_flesgKincaid+0.05*grade_numOfComplexConstructions+0.05*grade_fsw
    print('final grade - ',final_grade)
    print("---------------------------------------------------------------------")
    print('Token Grammar = ', len(encoding.encode(text+"Just check for punctuation: ")))
    print('Token AsessmentCreativety = ',len(encoding.encode(text+"Rate the creativity of the text from 0 to 10: ")))
    print('Token AsessmentSturcture = ',len(encoding.encode(text+"Evaluate structurality from 0 to 10: ")))
