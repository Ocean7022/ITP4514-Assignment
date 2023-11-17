import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

df = pd.read_json("../data/nbcnewsData.json")
df["text"] = df["title"] + ". " + df["content"]

self_words_list = pd.read_csv("../data/stopWordList.csv")["stop_word"]
all_stop_words = ENGLISH_STOP_WORDS.union(set(self_words_list))

tfidf = TfidfVectorizer(
    max_features=5000, stop_words=list(all_stop_words), token_pattern=r'\b[a-zA-Z]{2,}\b'
)
features_per_category = {}

for category in tqdm(
    df["category"].unique(), desc="Calculating TF-IDF", unit=" category"
):
    category_docs = df[df["category"] == category]["text"]
    tfidf_matrix = tfidf.fit_transform(category_docs)
    mean_scores = np.mean(tfidf_matrix, axis=0).A1
    features = tfidf.get_feature_names_out()
    scores_with_features = list(zip(mean_scores, features))
    sorted_scores = sorted(scores_with_features, reverse=True)
    features_per_category[category] = sorted_scores[:10]

for category in features_per_category:
    print(f"Category: {category}")
    for score, feature in features_per_category[category]:
        print(f"{feature}: {score}")
    print("\n")
