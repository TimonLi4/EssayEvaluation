import pandas as pd
import os
from dotenv import load_dotenv
import tiktoken
from agents import MAS


load_dotenv()


DATA_PATH = os.getenv('DATA_PATH')
encoding = tiktoken.encoding_for_model('gpt-4')


# key_api = os.getenv('API_KEY')


if __name__ =='__main__':
    df = pd.read_csv(DATA_PATH)
    text = df.iloc[10]['essay']
    final_score = 0

    mas = MAS(text)
    results = mas.evaluate()
    for result in results:
        print(result)
        final_score+=float(result['Grade'])*result['Weights']
    
    print('Final score: ',round(final_score,2))