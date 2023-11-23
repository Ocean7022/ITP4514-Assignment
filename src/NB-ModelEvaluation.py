import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import config.Config as config
from joblib import dump

print('Loading dataset...')
df = pd.read_json(config.dataSetPath)
df["text"] = df["title"] + ". " + df["content"]
print(len(df), 'data loaded from dataset')

# Vectorize news data
tfidf = config.tfidf
print('Vectorizing...')
X = tfidf.fit_transform(df['text'])
print('Vectorized')

# Save vectorizer
print('Saving vectorizer...')
dump(tfidf, config.vectorizerPath)
print('Vectorizer saved')

y = df['category']
# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

model = MultinomialNB()
print('Training model...')
model.fit(X_train, y_train)
print('Model trained')

# Save model
print('Saving model...')
dump(model, config.nb_classificationModelPath)
print('Model saved')

# make predictions for test data and print the classification report
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

