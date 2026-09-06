import pandas as pd
df = pd.read_csv("data.csv")
print(df) 
print(df.shape)
print(df['text'])
df['text_length'] = df['text'].str.len()
print(df)
positive_df = df[df['label'] == 'positive']
print(positive_df)
df.to_csv("processed_data.csv", index=False)
print("Data processing complete. Processed data saved to 'processed_data.csv'.")