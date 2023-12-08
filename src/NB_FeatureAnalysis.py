import numpy as np
from tqdm import tqdm
import pandas as pd
import config.Config as config

class NB_FeatureAnalysis:
    def __init__(self):
        self.numOfFeatures = input('\nHow many features do you want to output for each category? (10 - 50): ')
        if self.numOfFeatures.isdigit():
            self.numOfFeatures = int(self.numOfFeatures)
            if self.numOfFeatures < 10 or self.numOfFeatures > 50:
                print('Invalid input, please try again.')
                self.__init__()
        else:
            print('Invalid input, please try again.')
            self.__init__()

        self.__analyze()

    def __analyze(self):
        print('Loading dataset...')
        df = pd.read_json(config.dataSetPath)
        df["text"] = df["title"] + ". " + df["content"]
        print(len(df), 'data loaded from dataset')

        tfidf = config.tfidf
        features_per_category = {}

        for category in tqdm(
            df["category"].unique(), desc="Calculating TF-IDF", unit=" category"
        ):
            category_docs = df[df["category"] == category]["text"]
            tfidf_matrix = tfidf.fit_transform(category_docs)
            mean_scores = np.mean(tfidf_matrix, axis=0).A1
            features = tfidf.get_feature_names_out()
            scores_with_features = list(zip(mean_scores, features))
            sorted_scores = sorted(scores_with_features, reverse=True)
            features_per_category[category] = sorted_scores[:self.numOfFeatures]

        for category in features_per_category:
            print(f"\nCategory: {category}")
            for score, feature in features_per_category[category]:
                print(f"{feature}: {score}")
