from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # 從scikit-learn導入用於分割數據的工具
from sklearn.feature_extraction.text import TfidfVectorizer  # 從 scikit-learn 導入文本特徵提取工具
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # 導入英語的停用詞集
import pandas as pd  # 導入 pandas 庫，用於數據處理
from sklearn.metrics import classification_report, accuracy_score  # 導入評估模型性能的工具
import config.Config as config

print('Loading dataset...')
df = pd.read_json(config.dataSetPath)
df["text"] = df["title"] + ". " + df["content"]
print(len(df), 'data loaded from dataset')

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words=list(ENGLISH_STOP_WORDS.union(set(pd.read_csv(config.stopWordListPath)['stop_word']))),
    token_pattern=r'\b[a-zA-Z]{2,}\b'
)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)  # 分割數據為訓練集和測試集

rf = RandomForestClassifier(n_estimators=500, verbose=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)  # 使用模型進行預測
print(accuracy_score(y_test, y_pred))  # 計算並打印準確度
print(classification_report(y_test, y_pred))  # 輸出分類報告
