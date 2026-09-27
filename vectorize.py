import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# SPRINT 10.5: Expanding the Matrix (20 Words)
# ==========================================

# 1. Load the clean dataset
df = pd.read_csv("final_clean_dataset.csv")

# 2. The Ultimate Stopword Hitlist 
bengali_stopwords = ['এর', 'কী', 'বা', 'যে', 'হয়', 'করা', 'একটি', 'ও', 'কত', 'কোন', 'কোনটি', 'থেকে', 'হয়', 'কি', 'কীভাবে']

# 3. Initialize the TF-IDF Engine (Expanding to Top 20 features)
vectorizer = TfidfVectorizer(
    stop_words=bengali_stopwords, 
    max_features=20, 
    token_pattern=r'[\u0980-\u09FF]+'
)

# 4. The Detonator
tfidf_matrix = vectorizer.fit_transform(df['prompt_bn'].dropna())
important_words = vectorizer.get_feature_names_out()

print("\n--- The Top 20 Core Topics of Your Dataset ---")
for i, word in enumerate(important_words):
    print(f"Slot {i}: {word}")