from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from nltk.tokenize import sent_tokenize
import pandas as pd
import language_tool_python



tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")


def GrammarModel(input_text):
    """https://huggingface.co/grammarly/coedit-large"""
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=256)
    return outputs


def CheckGrammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches), matches



df = pd.read_csv(r'C:\Users\Timon\Desktop\Trainee\Automatic-Essay-Scoring-master\Processed_data.csv')
input_text = df.iloc[2]['essay']

sentences_list = sent_tokenize(input_text)


print(CheckGrammar(input_text))

# for index,i in enumerate(sentences_list):
#     outputs = GrammarModel(i)
#     edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f'{index}: 1st edition -- ', i, '2nd version -- ',edited_text)
#     print('same' if i == edited_text else 'False')  


