import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from bangla_stemmer.stemmer import stemmer
import os
import sys

# Block the stemmer's annoying "applied rules" spam in the terminal
sys.stdout = open(os.devnull, 'w')
bn_stemmer = stemmer.BanglaStemmer()
sys.stdout = sys.__stdout__

# ==========================================
# SPRINT 12: K-Means Clustering (Machine Learning)
# ==========================================

# 1. Load the clean dataset
df = pd.read_csv("final_clean_dataset.csv")

# 2. The Chopping Function (Machete)
def stem_sentence(sentence):
    words = sentence.split()
    # Suppress the print spam for each word
    sys.stdout = open(os.devnull, 'w')
    stemmed_words = [bn_stemmer.stem(word) for word in words]
    sys.stdout = sys.__stdout__
    return " ".join(stemmed_words)

print("1. Stemming the text...")
df['stemmed_prompt'] = df['prompt_bn'].astype(str).apply(stem_sentence)

# 3. The Ultimate Hitlist (Updated to catch the chopped stems!)
bengali_stopwords = ['এর', 'কী', 'বা', 'যে', 'হয়', 'করা', 'একটি', 'ও', 'কত', 'কোন', 'কোনটি', 'থেকে', 'হয়', 'কি', 'কীভাবে', 'থেক', 'কর', 'করে', 'হবে', 'হলে', 'এক', 'একা', 'পর']

# 4. Initialize TF-IDF Engine
print("2. Calculating the Math (TF-IDF)...")
vectorizer = TfidfVectorizer(
    stop_words=bengali_stopwords, 
    max_features=30, # Expanded to 30 to give the AI more context
    token_pattern=r'[\u0980-\u09FF]+'
)
tfidf_matrix = vectorizer.fit_transform(df['stemmed_prompt'])

# 5. UNLEASH THE AI: K-Means Clustering
print("3. Unleashing the AI Librarian (K-Means)...")
# We command the AI to find exactly 3 topics
kmeans = KMeans(n_clusters=3, random_state=42)
# The AI reads the math and groups the rows
df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

# 6. View the Results
print("\n=== AI CLUSTERING RESULTS ===")
for cluster_num in range(3):
    print(f"\n📁 FOLDER {cluster_num}:")
    # Grab 3 random sentences that the AI put in this folder
    sample_questions = df[df['Cluster'] == cluster_num]['prompt_bn'].sample(3, random_state=1)
    for q in sample_questions:
        print(f"  - {q}")