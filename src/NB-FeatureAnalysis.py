import numpy as np
import pandas as pd
from tqdm import tqdm
import config.Config as config

print("Loading dataset...")
df = pd.read_json(config.dataSetPath)
df["text"] = df["title"] + ". " + df["content"]
print(len(df), 'data loaded from dataset')

tfidf = config.tfidf
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
    print(f"\nCategory: {category}")
    for score, feature in features_per_category[category]:
        print(f"{feature}: {score}")
