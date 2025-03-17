import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset

# Download stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Load dataset
dataset = load_dataset("fancyzhx/ag_news", split="train[:5%]")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply preprocessing
df = pd.DataFrame(dataset)
df['text'] = df['text'].apply(preprocess_text)

# Save preprocessed data
df.to_csv("../results/cleaned_news.csv", index=False)
print("Preprocessed data saved to results/cleaned_news.csv")