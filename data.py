import jieba
import pandas as pd

df = pd.read_csv("train.csv", sep="\t", names=["label", "text"])

raw_texts = df["text"].tolist()
labels = df["label"].tolist()
processed_texts = [" ".join(jieba.lcut(str(text))) for text in raw_texts]

print(df.head())
