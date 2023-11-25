from joblib import load
import config.Config as config

print('Loading model...')
model = load(config.nb_classificationModelPath)
print('Model loaded successfully!')
print('Loading vectorizer...')
tfidf = load(config.vectorizerPath)
print('Vectorizer loaded successfully!')


with open('../data/testData/Test01-sport.txt', 'r') as f:
    testData = f.read()
X = tfidf.transform([testData])

print('Result is:', model.predict(X))

y_prob = model.predict_proba(X)

class_labels = model.classes_

for prob, label in zip(y_prob[0], class_labels):
    print(f"{label}: {prob:.4f}")
