from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.cluster import KMeans
import config.Config as config

# 從預訓練的bert-base-uncased模型載入分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 從預訓練的bert-base-uncased模型載入BERT模型
model = BertModel.from_pretrained('bert-base-uncased')

# 載入數據集
df = pd.read_json(config.dataSetPath)
# 將標題和內容拼接起來，中間加上句號和空格
df["text"] = df["title"] + ". " + df["content"]

# 取出數據集中的第四條新聞（從0開始計數）的文本
text_s = df['text'][3]
# 替換掉文本中的引號
text = text_s.replace('“', '').replace('”', '')

# 將文本按句號分割成多個句子
sentences = text.split('. ')

# 對每個句子進行BERT分詞，並創建attention mask
encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

is_cuda = torch.cuda.is_available()
if is_cuda:
    model = model.to('cuda')
    input_ids = input_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')
    print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else:
    print('Using CPU')

# 禁用梯度計算，進行模型的前向傳播，獲取輸出
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    # 取每個句子的[CLS]標記的輸出作為句子的嵌入表示
    sentence_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()

# 使用KMeans進行聚類分析，設置要形成的聚類數量
kmeans = KMeans(n_clusters=3, n_init=10)
# 對句子嵌入進行聚類
kmeans.fit(sentence_embeddings)

# 對每個聚類中心，找到最接近的句子
closest_sentences_indices = []
for cluster_center in kmeans.cluster_centers_:
    # 計算每個句子與聚類中心的距離
    distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
    # 找到距離最近的句子的索引
    closest_sentence_index = distances.argmin().item()
    closest_sentences_indices.append(closest_sentence_index)

# 確保選擇的句子是唯一的
unique_sentence_indices = list(set(closest_sentences_indices))

# 如果選擇的句子數量少於3個，則選擇剩餘最接近的句子
while len(unique_sentence_indices) < 3 and len(unique_sentence_indices) < len(sentences):
    for i, cluster_center in enumerate(kmeans.cluster_centers_):
        if len(unique_sentence_indices) == 3:
            break
        distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
        for index in distances.argsort():
            if index.item() not in unique_sentence_indices:
                unique_sentence_indices.append(index.item())
                break

# 根據索引提取摘要句子
summary_sentences = [sentences[idx] for idx in unique_sentence_indices if idx < len(sentences)]

# 將摘要句子組合成摘要
summary = '. '.join(summary_sentences)
print(summary + ".")
