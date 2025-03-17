import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
df = pd.read_csv("../results/cleaned_news.csv")

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
train_dataset = NewsDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = NewsDataset(val_texts.tolist(), val_labels.tolist())

# Model Training
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
training_args = TrainingArguments(
    output_dir="../results", eval_strategy="steps", eval_steps=50, per_device_train_batch_size=2,
    per_device_eval_batch_size=2, num_train_epochs=1, weight_decay=0.01, logging_dir="../logs", logging_steps=5
)
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
    compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}
)
trainer.train()

# Evaluation
preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
report = classification_report(val_labels, y_pred)

# Save report
with open("../results/classification_report.txt", "w") as f:
    f.write(report)
print("Classification report saved to results/classification_report.txt")