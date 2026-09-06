import pandas as pd

# 1. Ingestion
df = pd.read_csv("test set.csv")
print("Original matrix dimensions:", df.shape)

# 2. Feature Engineering
df['prompt_length'] = df['prompt_bn'].str.len()

# 3. Filtering
null_context_df = df[df['context'] == '[NULL]']

# 4. Export
null_context_df.to_csv("processed_test_set.csv", index=False)
print("Filtered matrix dimensions:", null_context_df.shape)