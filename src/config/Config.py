import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# read data set
def readDataSet():
    print('Reading data set...')
    newsDataSet = []
    for i in range(4):
        with open(f'../data/newsDataSet-part{i + 1}.json', 'r') as f:
            newsDataSet.append(pd.read_json(f))
    combined_data = pd.concat(newsDataSet, ignore_index=True)
    print('Data set length:', len(combined_data))
    return combined_data
        

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
