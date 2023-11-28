import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import json
import config.Config as config
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 加载数据集
with open(config.dataSetPath, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取文本和标签
texts = [item['title'] + '. ' + item['content'] for item in data]
labels = [item['category'] for item in data]

# 下载停用词
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 实例化词干提取器
stemmer = PorterStemmer()

def clean_text(text):
    # 去除标点
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 分词
    words = word_tokenize(text.lower())
    # 去除停用词并进行词干提取
    return [stemmer.stem(word) for word in words if word not in stop_words]

# 应用数据清洗
tokenized_texts = [clean_text(text) for text in tqdm(texts, desc='Cleaning and Tokenizing', ncols=100)]

# 构建词频表
word_freq = Counter(word for sentence in tqdm(tokenized_texts, desc='Processing Texts', ncols=100) for word in sentence)

# 取最常见的vocab_size-1个单词加上一个UNK（未知词
vocab = [word for word, freq in tqdm(word_freq.most_common(config.vocab_size - 1), desc='Creating Vocabulary', ncols=100)]
vocab.append("UNK")

# 创建单词到索引的映射
word_to_index = {word: index for index, word in enumerate(vocab)}

# 转换文本为索引序列
text_sequences = [[word_to_index.get(word, word_to_index["UNK"]) for word in text] for text in tqdm(tokenized_texts, desc='Converting to Sequences', ncols=100)]

# 标签编码
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 创建自定义Dataset
class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 对文本序列进行填充或截断
padded_sequences = pad_sequence([torch.tensor(seq[:config.max_length]) for seq in tqdm(text_sequences, desc='Padding Sequences', ncols=100)], batch_first=True)

# 实例化Dataset
dataset = NewsDataset(padded_sequences, torch.tensor(encoded_labels))

# 数据集分割
train_size = int(0.8 * len(dataset))  # 假设训练集占80%
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 设置设备
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else:
    device = torch.device('cpu')
    print('Using CPU')

# 实例化模型
model = RNN(config.input_size, config.hidden_size, config.num_layers, len(set(labels))).to(device)

# 损失函数和优化器
num_samples_class = [1888, 17497, 1020, 623, 2134, 565, 1184, 693, 16004]
weights = [1 - (x / 41608) for x in num_samples_class]
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)


best_val_loss = float('inf')
patience = 5
patience_counter = 0
correct_predictions = 0
total_predictions = 0

# 训练模型
print('Start training...')
for epoch in range(config.num_epochs):
    model.train()
    for i, (texts, labels) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{config.num_epochs}', total=len(train_loader), ncols=100):
        texts = texts.to(device)
        labels = labels.to(device)

        # 进行词嵌入
        embedded_texts = embedding(texts)

        # 前向传播和后向传播
        outputs = model(embedded_texts)
        loss = criterion(outputs, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            embedded_texts = embedding(texts)
            outputs = model(embedded_texts)

            # 计算损失
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {avg_val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%')

    # 检查是否应该执行早停
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # 可选：保存当前最佳模型
        #torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break
print('Finished Training')

# 测试模型
print('Start testing...')
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_loader:
        texts = texts.to(device)
        labels = labels.to(device)
        
        embedded_texts = embedding(texts)
        
        outputs = model(embedded_texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
print('Finished Testing')
