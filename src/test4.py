import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import config.Config as config

# 准备数据
def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

# 文本预处理
def preprocess(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    return ' '.join(tokens)

# 加载数据
df = load_data(config.dataSetPath)

# 预处理内容
df['processed_content'] = df['content'].apply(preprocess)

# 特征提取
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_content'])
y = df['category']

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 模型训练
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# 模型评估
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
