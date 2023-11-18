import numpy as np  # 導入 NumPy 庫，用於數值計算
import pandas as pd  # 導入 pandas 庫，用於數據處理
from sklearn.feature_extraction.text import TfidfVectorizer  # 從 scikit-learn 導入文本特徵提取工具
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # 導入英語的停用詞集
from tqdm import tqdm  # 導入 tqdm 庫，用於顯示進度條
import config.nbConfig as config

df = pd.read_json(config.dataSetPath)  # 從JSON文件加載數據到pandas DataFrame
df["text"] = df["title"] + ". " + df["content"]  # 將新聞的標題和內容結合，形成一個新的列 'text'

self_words_list = pd.read_csv(config.stopWordListPath)["stop_word"]  # 從CSV文件加載自定義停用詞列表
all_stop_words = ENGLISH_STOP_WORDS.union(set(self_words_list))  # 結合 scikit-learn 的英文停用詞和自定義停用詞

tfidf = TfidfVectorizer(
    #max_features=5000, stop_words=list(all_stop_words), token_pattern=r'\b[a-zA-Z]{2,}\b'
)  # 初始化 TF-IDF 向量化器，設置最大特徵數量、停用詞和令牌模式

features_per_category = {}  # 創建一個空字典，用於存儲每個類別的特徵

for category in tqdm(
    df["category"].unique(), desc="Calculating TF-IDF", unit=" category"
):  # 對每個獨特的類別進行迴圈，並顯示進度條
    category_docs = df[df["category"] == category]["text"]  # 從 DataFrame 中選取該類別的所有文檔
    tfidf_matrix = tfidf.fit_transform(category_docs)  # 對這些文檔應用 TF-IDF 轉換
    mean_scores = np.mean(tfidf_matrix, axis=0).A1  # 計算每個特徵的平均 TF-IDF 分數
    features = tfidf.get_feature_names_out()  # 獲取特徵名稱
    scores_with_features = list(zip(mean_scores, features))  # 將特徵分數與特徵名稱配對
    sorted_scores = sorted(scores_with_features, reverse=True)  # 對特徵分數進行降序排序
    features_per_category[category] = sorted_scores[:10]  # 為每個類別存儲前 10 個最高分數的特徵

for category in features_per_category:  # 遍歷每個類別
    print(f"Category: {category}")  # 打印類別名稱
    for score, feature in features_per_category[category]:  # 遍歷該類別的前 10 個特徵
        print(f"{feature}: {score}")  # 打印特徵名稱和對應的分數
    print("\n")  # 打印一個空行，用於分隔不同的類別
