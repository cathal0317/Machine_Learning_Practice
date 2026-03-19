from Programming_AI.ai504_03_pytorch import train_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch import nn
from transformers import Trainer
from transformers import TrainingArguements
from transformers import pipeline
from sklearn.metrics import f1_score
import numpy as np
import torch

# Call Model, initialise tokenizer
model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

emotion_dataset = load_dataset("emotion")

# print(emotion_dataset)

data_df = emotion_dataset["train"].to_pandas()
print(data_df.head())

features = emotion_dataset["train"].features
print(features)

id2label = {idx:features["label"].int2str(idx) for idx in range(len(features["label"].names))}
print(id2label)

label2id = {v:k for k, v in id2label.items()}
print(label2id)

print(data_df["label"].value_counts(normalize=True).sort_index())

def tokenised_inputs(inputs):
    return tokenizer(inputs["text"], truncation=True, max_length=512)

emotion_dataset = emotion_dataset.map(tokenised_inputs, batched=True)

print(emotion_dataset)

# mnist_dataset = load_dataset("ylecun/mnist")
# print(mnist_dataset)

# feature = mnist_dataset["train"].features
# print(feature)


# Deal with unbalanced datasets

class_weights = (1 - (data_df["label"].value_counts().sort_index() / len(data_df))).values
print(class_weights)

# Rename label to feed into trainer later
emotion_dataset = emotion_dataset.rename_column("label", "labels")

# Define own trainer
class Weighted_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_Outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = input.get("labels")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_Outputs else loss

# Define model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id)

# Compute f1 score for unbalanced data

def compute_metric(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average = "weighted")
    return {"f1" : f1}

batch_size = 64

logging_steps = len(emotion_dataset["train"]) // batch_size
output_dir = "minilm_finetuned_emotion"
training_args = TrainingArguements(output_dir=output_dir, num_train_epochs=5,learning_rate=2e-5,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,weight_decay=0.01,evaluation_strategy="epoch", logging_steps=logging_steps,fp16=True,push_to_hub=True)

trainer = Weighted_Trainer(model=model, args=training_args, compute_metrics=compute_metrics, train_dataset=emotion_dataset["train"], eval_dataset=emotion_dataset["validation"],tokenizer=tokenizer)

trainer.train()