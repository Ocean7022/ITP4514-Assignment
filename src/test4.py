from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def summarize_text(text, num_sentences=5):
    sentences = text.split('.')
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    sentence_scores = cosine_matrix.sum(axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[::-1]]
    summary = '. '.join(ranked_sentences[:num_sentences])
    return summary

df = pd.read_json("./data/dataset.json")
df["text"] = df["title"] + ". " + df["content"]
article_text = df["text"][0]
summary = summarize_text(article_text, num_sentences=2)
print(summary)
