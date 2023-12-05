import string, re
import config.Config as config
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class GRU_DataProcess:
    def __init__(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = word_tokenize(text.lower())
        pattern = re.compile(config.pattern)
        text = [word for word in text if pattern.match(word)]
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if word not in config.stopWordList]