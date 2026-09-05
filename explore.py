import pandas as pd
df = pd.read_csv("data.csv")
print(df) 
print(df.shape)
print(df['text'])
df['text_length'] = df['text'].str.len()
print(df)