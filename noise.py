import random
import re
import pandas as pd

NOISE = [".", "*", "-", "_", "/", "|"]


def inject_noise(text, prob=0.15):
    new_text = []
    for ch in text:
        new_text.append(ch)
        if random.random() < prob:
            new_text.append(random.choice(NOISE))
    return "".join(new_text)


aug_texts = []
aug_labels = []

df = pd.read_csv("train.csv", sep=",", header=0, names=["label", "text"])
df.dropna(inplace=True)
print(df[~df["label"].apply(lambda x: str(x).isdigit())])
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)
print(df[~df["label"].apply(lambda x: str(x).isdigit())])
for text, label in zip(df["text"], df["label"]):
    aug_texts.append(text)
    aug_labels.append(label)

    if label == 1:
        for _ in range(2):
            aug_texts.append(inject_noise(text))
            aug_labels.append(1)

aug_df = pd.DataFrame({"label": aug_labels, "text": aug_texts})
aug_df.to_csv("train_augmented.csv", index=False, header=False)
print(f"原始数据: {len(df)} 条")
print(f"增强后数据: {len(aug_df)} 条")
print(f'广告样本从 {sum(df["label"] == 1)} 增加到 {sum(aug_df["label"] == 1)}')
