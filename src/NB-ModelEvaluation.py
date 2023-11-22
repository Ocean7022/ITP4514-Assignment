import pandas as pd  # 導入pandas庫，用於數據操作
from sklearn.model_selection import train_test_split  # 從scikit-learn導入用於分割數據的工具
from sklearn.naive_bayes import MultinomialNB  # 導入多項式朴素貝葉斯分類器
from sklearn.metrics import classification_report, accuracy_score  # 導入評估模型性能的工具
import config.Config as config
from joblib import dump

print('Loading dataset...')
df = pd.read_json(config.dataSetPath)
df["text"] = df["title"] + ". " + df["content"]
print(len(df), 'data loaded from dataset')

tfidf = config.tfidf
X = tfidf.fit_transform(df['text'])  # 將文本數據轉換為TF-IDF特徵
dump(tfidf, config.vectorizerPath) # 儲存tfidf向量器
y = df['category']  # 設置目標變量為‘category’列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)  # 分割數據為訓練集和測試集

model = MultinomialNB()  # 初始化多項式朴素貝葉斯模型
model.fit(X_train, y_train)  # 使用訓練數據訓練模型

y_pred = model.predict(X_test)  # 使用模型進行預測
print(accuracy_score(y_test, y_pred))  # 計算並打印準確度
print(classification_report(y_test, y_pred))  # 輸出分類報告

dump(model, './model/NB-Model.joblib')
