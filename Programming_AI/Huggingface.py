from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import torch

emotion_dataset = load_dataset("emotion")
emotion 
def print_encoding(model_inputs, indent=4):
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")

tokeniser = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")


model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

inputs = "I'm excited to learn about Huggin Face Transformers!"

tokenised_inputs = tokeniser(inputs, return_tensors = "pt")
outputs = model(**tokenised_inputs)

labels = ['NEGATIVE', 'POSITIVE']
prediction = torch.argmax(outputs.logits)


print("Input:")
print(inputs)
print()
print("Tokenized Inputs:")
print_encoding(tokenised_inputs)
print()
print("Model Outputs:")
print(outputs)
print()
print(f"The prediction is {labels[prediction]}")

def tokenised_text(examples):
    return tokeniser(examples["text"], truncation=True, max_length=512 )

