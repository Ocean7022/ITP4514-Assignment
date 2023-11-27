import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# classifier
dataSetPath = '../data/newsDataSet.json'

#tfidf
stopWordListPath = '../data/stopWordList.csv'
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words=list(ENGLISH_STOP_WORDS.union(set(pd.read_csv(stopWordListPath)['stop_word']))),
    token_pattern=r'\b[a-zA-Z]{2,}\b'
)
#tfidf = TfidfVectorizer()
vectorizerPath = './model/tfidf_vectorizer.joblib'

# NB model
nb_classificationModelPath = './model/NB-Model.joblib'
