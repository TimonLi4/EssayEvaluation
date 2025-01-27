from transformers import pipeline


text_analyzer = pipeline('text-classification',model = 'bert-base-uncased')

text = "Our friends won't buy this analysis, let alone the next one we propose."

# Анализ текста
result = text_analyzer(text)
print(result)