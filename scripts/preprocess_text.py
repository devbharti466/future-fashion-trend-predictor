import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")

df = pd.read_csv("data/fashion_trends.csv")

def clean_text(text):
    text = re.sub(r"\W", " ", text) 
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

df["Processed_Description"] = df["Description"].apply(clean_text)

df.to_csv("data/processed_fashion_trends.csv", index=False)
print("Text Preprocessing Complete!")
