import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_json("./data/dataset.json")
df["text"] = df["title"] + ". " + df["content"]

self_words_list = pd.read_csv("./data/stopWordList.csv")["stop_word"]
all_stop_words = ENGLISH_STOP_WORDS.union(set(self_words_list))

tfidf = TfidfVectorizer(
    max_features=5000, stop_words=list(all_stop_words), token_pattern=r'\b[a-zA-Z]{2,}\b'
)
X = tfidf.fit_transform(df['text'])
y = df['category']

print(tfidf.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
