import pandas as pd
import ollama
import tiktoken


encoding = tiktoken.encoding_for_model('gpt-4')


def CheckGrammar(text):
    """function checks grammar and punctuation. LLM problem"""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"just check for punctuation: {text}"}])
    return response["message"]["content"]


def AsessmentCreativety(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"rate the creativity of the text from 0 to 10: {text}"}])
    return response["message"]["content"]


def AsessmentSturcture(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Evaluate structurality from 0 to 10: {text}"}])
    return response["message"]["content"]

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Timon4\Desktop\projectTrainee\other\Automatic-Essay-Scoring-master\Processed_data.csv')
    text = df.iloc[0]['essay']
    # print(text)
    grammar = CheckGrammar(text)
    creativety = AsessmentCreativety(text)
    structure = AsessmentSturcture(text)
    
    print('--------------------------------------------------------------------------------')
    print(grammar)
    print('----------')
    print(creativety)
    print('----------')
    print(structure)

    print('--------------------------------------------------------------------------------')
    print('\n')
    print('\n')
    print(f'Token Grammar = ', len(encoding.encode(text+"Just check for punctuation: ")))
    print(f'Token AsessmentCreativety = ',len(encoding.encode(text+"Rate the creativity of the text from 0 to 10: ")))
    print(f'Token AsessmentSturcture = ',len(encoding.encode(text+"Evaluate structurality from 0 to 10: ")))