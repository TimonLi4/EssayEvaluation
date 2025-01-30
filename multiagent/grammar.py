import pandas as pd
import ollama



def CheckGrammar(text):
    """function checks grammar and punctuation. LLM problem"""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Score 0 to 10 on grammar and punctuation. just a number: {text}"}])
    return response["message"]["content"]


def AsessmentCreativety(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"rate the creativity of the text from 0 to 10. Write just a number: {text}"}])
    return response["message"]["content"]


def AsessmentSturcture(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Evaluate structurality from 0 to 10. Write just a number: {text}"}])
    return response["message"]["content"]

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Timon4\Desktop\projectTrainee\other\Automatic-Essay-Scoring-master\Processed_data.csv')
    text = df.iloc[0]['essay']
    # print(text)
    