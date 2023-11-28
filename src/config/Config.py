import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# DataSet
dataSetPath = '../data/newsDataSet.json'
        
#tfidf
stopWordListPath = '../data/stopWordList.csv'
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    stop_words=list(ENGLISH_STOP_WORDS.union(set(pd.read_csv(stopWordListPath)['stop_word']))),
    token_pattern=r'\b[a-zA-Z]{2,}\b',
    max_df=0.5,
    min_df=3,
    norm='l2',
    sublinear_tf=True
)
#tfidf = TfidfVectorizer()
vectorizerPath = './model/tfidf_vectorizer.joblib'

# NB model
nb_classificationModelPath = './model/NB-Model.joblib'



# RNN model
max_length = 300

vocab_size = 7000
embedding_dim = 150

input_size = embedding_dim
hidden_size = 256
num_layers = 4

learning_rate = 0.001
num_epochs = 20

