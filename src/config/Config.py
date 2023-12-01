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
dataSetPath01 = '../data/RNNDataSet01.pth'
dataSetPath02 = '../data/RNNDataSet02.pth'
RNNDataSetPath = dataSetPath01

stopWordList = ENGLISH_STOP_WORDS.union(set(pd.read_csv(stopWordListPath)['stop_word']))
pattern = r'\b[a-zA-Z]{3,}\b'

batch_size = 128
train_ratio = 0.8

max_length = 900
vocab_size = 73000
embedding_dim = 600
input_size = embedding_dim
hidden_size = 128
num_layers = 2
num_classes = 9 # num of types of news
learning_rate = 0.005
num_epochs = 10

