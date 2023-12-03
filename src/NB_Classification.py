from joblib import load
import config.Config as config
import os

class NB_Classification:
    def __init__(self, testData):
        self.testData = testData['file_content']
        self.__classify()

    def __classify(self):
        print('\nLoading model...')
        model = load(config.nb_classificationModelPath)
        print('Model loaded successfully!')
        print('Loading vectorizer...')
        tfidf = load(config.vectorizerPath)
        print('Vectorizer loaded successfully!')

        X = tfidf.transform([self.testData])

        print('\nResult is:', model.predict(X))

        return

        y_prob = model.predict_proba(X)
        class_labels = model.classes_

        for prob, label in zip(y_prob[0], class_labels):
            print(f"{label}: {prob:.4f}")
