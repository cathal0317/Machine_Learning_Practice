from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import torch

mnist_dataset = load_dataset("ylecun/mnist")
print(mnist_dataset)

feature = mnist_dataset["train"].features
print(feature)

panda_mnist = mnist_dataset["train"].to_pandas()
print(panda_mnist.head())

data_df = panda_mnist['label'].value_counts(normalize=True).sort_index()
print(data_df)