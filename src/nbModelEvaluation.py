import pandas as pd  # 導入pandas庫，用於數據操作
from sklearn.model_selection import train_test_split  # 從scikit-learn導入用於分割數據的工具
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS  # 導入用於文本特徵提取的工具
from sklearn.naive_bayes import MultinomialNB  # 導入多項式朴素貝葉斯分類器
from sklearn.metrics import classification_report, accuracy_score  # 導入評估模型性能的工具

df = pd.read_json("../data/nbcnewsData.json")  # 從JSON文件加載數據到pandas DataFrame
df["text"] = df["title"] + ". " + df["content"]  # 結合標題和內容到一個新的列‘text’

self_words_list = pd.read_csv("../data/stopWordList.csv")["stop_word"]  # 從CSV文件加載自定義停用詞列表
all_stop_words = ENGLISH_STOP_WORDS.union(set(self_words_list))  # 結合scikit-learn的英語停用詞和自定義停用詞

tfidf = TfidfVectorizer(
    max_features=5000, stop_words=list(all_stop_words), token_pattern=r'\b[a-zA-Z]{2,}\b'
)  # 初始化TF-IDF向量化器，設置最大特徵數，停用詞和令牌模式
X = tfidf.fit_transform(df['text'])  # 將文本數據轉換為TF-IDF特徵
y = df['category']  # 設置目標變量為‘category’列

print(tfidf.get_feature_names_out())  # 輸出特徵名稱

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 分割數據為訓練集和測試集

model = MultinomialNB()  # 初始化多項式朴素貝葉斯模型
model.fit(X_train, y_train)  # 使用訓練數據訓練模型

y_pred = model.predict(X_test)  # 使用模型進行預測
print(accuracy_score(y_test, y_pred))  # 計算並打印準確度
print(classification_report(y_test, y_pred))  # 輸出分類報告