import pandas as pd
import matplotlib.pyplot as plt
import string

# Ingestion
df = pd.read_csv("processed_test_set.csv")

# Proof of life
print(df.head())

# 1. Command pandas to calculate the geometry in RAM
df['prompt_length'].plot(kind='hist', bins = 20, title='Prompt Lengths', color = 'blue')

# 2. Command matplotlib to export the final rendering
plt.savefig('length_chart.png')
print("Chart successfully rendered and saved!")

# 3. Data Cleaning: The Boolean Mask
clean_df = df[~df['prompt_bn'].str.contains('svg', na=False)]

# 4. Final Export
clean_df.to_csv("final_clean_dataset.csv", index=False)
print(f"Cleaned dataset ready! Total rows: {len(clean_df)}")

# 5. Isolate a single Bengali string from the clean dataset
sample_sentence = clean_df['prompt_bn'].iloc[1]
print("--- Raw String ---")
print(sample_sentence)

# 6. Basic Tokenization (Chopping by spaces)
tokens = sample_sentence.split()
print("\n--- Tokens ---")
print(tokens)
print("Total Tokens: ", len(tokens))

# 7. Advanced Tokenization (Targeted Punctuation Scrubbing)
# Combine all standard punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) with the Bengali dari
target_punctuation = string.punctuation + '।'

# Command the processor to translate all target punctuation into absolute nothingness
translator = str.maketrans('', '', target_punctuation)
sanitized_sentence = sample_sentence.translate(translator)
print("\n--- Sanitized String ---")
print(sanitized_sentence)

final_tokens = sanitized_sentence.split()
print("\n--- Final Tokens ---")
print(final_tokens)

# 8. Scale to the Full Matrix
# Create a safe independent copy of the matrix to prevent pandas memory warnings
clean_df = clean_df.copy() 

# Fire the translator logic across the entire prompt column instantly
clean_df['sanitized_prompt'] = clean_df['prompt_bn'].apply(lambda x: str(x).translate(translator))

# Chop every sanitized sentence into a list of tokens
clean_df['tokens'] = clean_df['sanitized_prompt'].str.split()

print("\n--- Full Matrix Tokenization ---")
print(clean_df[['prompt_bn', 'tokens']].head())