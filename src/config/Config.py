import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# classifier
dataSetPath = '../data/newsDataSet.json'

#tfidf
stopWordListPath = '../data/stopWordList.csv'
max_features = 5000
token_pattern = r'\b[a-zA-Z]{2,}\b'
tfidf = TfidfVectorizer(
    max_features=max_features,
    stop_words=list(ENGLISH_STOP_WORDS.union(set(pd.read_csv(stopWordListPath)['stop_word']))),
    token_pattern=token_pattern
)
vectorizerPath = './model/tfidf_vectorizer.joblib'

# model
nb_classificationModelPath = './model/NB-Model.joblib'

apiKey = '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'