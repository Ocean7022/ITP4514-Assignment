import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

stopWordListPath = '../data/stopWordList.csv'
testDataFolderPath = '../data/testData'

# NB model
dataSetPath = '../data/newsDataSet.json'
vectorizerPath = './model/tfidf_vectorizer.joblib'
nb_classificationModelPath = './model/NB-ClassificationModel.joblib'
resultPhotoPath = '../img/NB-Tuning/Result.png'
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

# GUR model
GRUDataSetPath = '../data/GRUDataSet.pth'
GRU_classificationModelPath = './model/GRU-ClassificationModel.pth'

stopWordList = ENGLISH_STOP_WORDS.union(set(pd.read_csv(stopWordListPath)['stop_word']))
pattern = r'\b[a-zA-Z]{3,}\b'

batch_size = 128
train_ratio = 0.8

# Data preprocessing parameters
max_length = 1000
vocab_size = 14000
embedding_dim = 400
input_size = embedding_dim

# Training model parameters
hidden_size = 128
num_layers = 3
num_classes = 9 # num of types of news
learning_rate = 0.003
num_epochs = 20

