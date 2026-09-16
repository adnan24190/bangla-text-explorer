import pandas as pd
import matplotlib.pyplot as plt
import string

# Ingestion
df = pd.read_csv("processed_test_set.csv")

# Proof of life
print(df.head())

# 1. Command pandas to calculate the geometry in RAM
df['prompt_length'].plot(kind='hist', bins=20, title='Prompt Length', color='green')

# 2. Command matplotlib to export the final rendering
plt.savefig('length_chart.png')
print("Length chart successfully rendered and saved!")

# 3. Data Cleaning: The Boolean Mask
clean_df = df[~df['prompt_bn'].str.contains('svg', na = False)]

# 4. Final Export
clean_df.to_csv("final_clean_dataset.csv", index=False)
print(f"Cleaned dataset ready!  Total Rows = {len(clean_df)}")


# 5. (Targeted Punctuation Scrubbing)
# Combine all standard punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~) with the Bengali dari
target_punctuation = string.punctuation + '।'

# Command the processor to translate all target punctuation into absolute nothingness
translator = str.maketrans('', '', target_punctuation)


# 6. Scale to the Full Matrix
# Create a safe independent copy of the matrix to prevent pandas memory warnings
clean_df = clean_df.copy() 

# Fire the translator logic across the entire prompt column instantly
clean_df['sanitized_prompt'] = clean_df['prompt_bn'].apply(lambda x: str(x).translate(translator))

# Chop every sanitized sentence into a list of tokens
clean_df['tokens'] = clean_df['sanitized_prompt'].str.split()

print("\n--- Full Matrix Tokenization ---")
print(clean_df[['prompt_bn', 'tokens']].head())

# ==========================================
# SPRINT 9: NLP Stopword Filtering
# ==========================================

# 1. Define the useless filler words (The Stopword Hitlist)
bengali_stopwords = ['এর', 'কী', 'বা', 'যে', 'হয়', 'করা']

# 2. The Filter (List Comprehension + Scattergun)
# Command the CPU to keep a word ONLY if it is not in the stopword list
clean_df['clean_tokens'] = clean_df['tokens'].apply(lambda token_list: [word for word in token_list if word not in bengali_stopwords])

print("\n--- After Stopword Removal ---")
print(clean_df[['prompt_bn', 'clean_tokens']].head())

# ==========================================
# SPRINT 9.5: Frequency Analysis (The Terminal Bypass)
# ==========================================

# 1. The Detonator: Blow open the lists and flatten them into a single column
all_words = clean_df['clean_tokens'].explode()

# 2. The Accountant: Command pandas to count how many times each word appears
word_frequencies = all_words.value_counts()

# 3. THE BYPASS: Export the exact counts directly to a CSV file
word_frequencies.to_csv("word_frequencies.csv", encoding='utf-8-sig')

print("\n--- Frequency Analysis Complete ---")
print("Terminal rendering bypassed! Check your folder for 'word_frequencies.csv'.")