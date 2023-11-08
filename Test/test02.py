import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 读取CSV文件
df = pd.read_csv('dataset.csv')

# 可以选择合并'title'和'content'列作为特征
# 这样做的好处是标题中可能包含对分类有帮助的关键词
df['text'] = df['title'] + " " + df['content']

# 文本预处理（此处省略具体步骤，您需要根据需要添加）

# 特征提取
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['category']  # 注意这里的列名改为'category'

print(tfidf.get_feature_names_out())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
