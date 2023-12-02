from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import config.Config as config
from joblib import dump
import pandas as pd
import time
import matplotlib.pyplot as plt

class NB_ModelEvaluation:
    def __init__(self):
        while True:
            selection = input('\nDo you want to save the vectorizer and model? (y/n): ')
            if selection == 'y':
                self.save = True
                break
            elif selection == 'n':
                self.save = False
                break
            else:
                print('Invalid input, please try again.')

        while True:
            selection = input('Do you want to output the evaluation report as a photo? (y/n): ')
            if selection == 'y':
                self.isOutputPhoto = True
                break
            elif selection == 'n':
                self.isOutputPhoto = False
                break
            else:
                print('Invalid input, please try again.')

        self.__evaluate()

    def __evaluate(self):
        print('Loading dataset...')
        df = pd.read_json(config.dataSetPath)
        df["text"] = df["title"] + ". " + df["content"]
        print(len(df), 'data loaded from dataset')

        # Vectorize news data
        tfidf = config.tfidf
        print('Vectorizing...')
        startTime = time.time()
        X = tfidf.fit_transform(df['text'])
        endTime = time.time()
        print('Vectorized', len(df), 'news in', endTime - startTime, 'seconds')

        # Save vectorizer
        if self.save:
            print('Saving vectorizer...')
            dump(tfidf, config.vectorizerPath)
            print('Vectorizer saved')

        y = df['category']
        # split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

        model = MultinomialNB()
        print('Training model...')
        model.fit(X_train, y_train)
        print('Model trained')

        # Save model
        if self.save:
            print('Saving model...')
            dump(model, config.nb_classificationModelPath)
            print('Model saved')

        # make predictions for test data
        y_pred = model.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        print('\nAccuracy:', accuracy)
        print(classification_report(y_test, y_pred))

        if self.isOutputPhoto:
            self.outputPhoto(y_test, y_pred)

    def outputPhoto(self, y_test, y_pred):
        report = classification_report(y_test, y_pred, output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])

        precisions = report_df['precision'].tolist()
        recalls = report_df['recall'].tolist()
        f1_scores = report_df['f1-score'].tolist()
        categories = report_df.index.tolist()

        bar_width = 0.25
        index = range(len(categories))

        fig, ax = plt.subplots(figsize=(11, 7))
        bar1 = ax.bar([i - bar_width for i in index], precisions, width=bar_width, label='Precision')
        bar2 = ax.bar(index, recalls, width=bar_width, label='Recall')
        bar3 = ax.bar([i + bar_width for i in index], f1_scores, width=bar_width, label='F1-Score')

        ax.set_xlabel('Category')
        ax.set_ylabel('Scores')
        ax.set_title('Precision, Recall and F1-Score for Each Category')
        ax.set_xticks([i for i in index])
        ax.set_xticklabels(categories, rotation=45)
        ax.legend()
        
        plt.subplots_adjust(bottom=0.17, top=0.94, left=0.06, right=0.99)

        plt.savefig(config.resultPhotoPath)
        print(f'Evaluation report saved as [ {config.resultPhotoPath} ]')
