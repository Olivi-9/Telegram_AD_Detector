import pandas as pd

df = pd.read_csv("spanish_data.csv")
df = df[["label", "text"]]  # 显式重排列顺序
df.to_csv("changed_spanish_data.csv", index=False)
