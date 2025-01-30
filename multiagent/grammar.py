import pandas as pd
import ollama



def CheckGrammar(text):
    """function checks grammar and punctuation. LLM problem"""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": "Rate the grammar and punctuation of the following text on a scale from 0 to 10, where 0 means full of errors and 10 means perfect. Output only the number without any additional comments:{text}"}])
    return response["message"]["content"]


def AsessmentCreativety(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of creativity of the following text on a scale of 0 to 10, where 0 is completely uncreative and 10 is maximally creative. Print only the number without any additional comments:{text}"}])
    return response["message"]["content"]


def AsessmentSturcture(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of structuredness of the following text on a scale from 0 to 10, where 0 means completely chaotic and 10 means perfectly structured. Output only the number without any additional comments:{text}"}])
    return response["message"]["content"]

def Information(text):
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": f"Rate the level of informativeness of the following text on a scale from 0 to 10, where 0 means completely uninformative and 10 means highly informative. Output only the number without any additional comments:{text}"}])
    return response["message"]["content"]

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\Timon4\Desktop\projectTrainee\other\Automatic-Essay-Scoring-master\Processed_data.csv')
    text = df.iloc[0]['essay']
    # print(text)
    