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
import string, os, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    
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
    
# get dataSet
def dataPorcess():
    if os.path.exists(config.RNNDataSetPath):
        print('Processed DataSet already exists.')
        return getTrainAndTestLoder(torch.load(config.RNNDataSetPath))
    else:
        print('Processed DataSet does not exist, start processing...')
        print('Reading DataSet...')
        with open(config.dataSetPath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print('DataSet read successfully.', len(data), 'items in total.')

        nltk.download('punkt')

        texts = [item['title'] + '. ' + item['content'] for item in data]
        labels = [item['category'] for item in data]
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        translator = str.maketrans('', '', string.punctuation)
        texts = [text.translate(translator) for text in tqdm(texts, desc='Removing Punctuations', ncols=100, unit='item')]

        texts = [word_tokenize(text.lower()) for text in tqdm(texts, desc='Tokenizing', ncols=100, unit='item')]


        #pattern = re.compile(config.pattern)
        #texts = [[word for word in text if pattern.match(word)] for text in tqdm(texts, desc='Removing Non-English Words', ncols=100, unit='item')]

        #texts = [[word for word in text if word not in config.stopWordList] for text in tqdm(texts, desc='Removing Stopwords', ncols=100, unit='item')]

        #stemmer = PorterStemmer()
        #texts = [[stemmer.stem(word) for word in text] for text in tqdm(texts, desc='Stemming', ncols=100, unit='item')]

        all_words = [word for text in texts for word in text]
        word_freq = Counter(all_words)
        vocab = {"UNK": 0}
        vocab.update({word: idx + 1 for idx, (word, _) in enumerate(word_freq.items())})

        word_to_index = {word: index for index, word in enumerate(vocab)}
        texts = [[word_to_index.get(word, word_to_index["UNK"]) for word in text] for text in tqdm(texts, desc='Converting to Sequences', ncols=100)]

        padded_sequences = pad_sequence([torch.tensor(seq[:config.max_length]) for seq in tqdm(texts, desc='Padding Sequences', ncols=100)], batch_first=True)
        
        dataset = NewsDataset(padded_sequences, torch.tensor(labels))
        torch.save(dataset, config.RNNDataSetPath)
        return getTrainAndTestLoder(dataset)
    
def getTrainAndTestLoder(dataset):
    train_size = int(config.train_ratio * len(dataset))
    test_size = int((1 - config.train_ratio) * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def getDevice():
    if torch.cuda.is_available():
        print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        return torch.device('cuda')
    else:
        print('Using CPU')
        return torch.device('cpu')

def countAvgLength(texts):
    total = 0
    for text in texts:
        total += len(text)
    return total / len(texts)

device = getDevice()
train_loader, test_loader = dataPorcess()
model = RNN(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)

num_samples_class = [1888, 17497, 1020, 623, 2134, 565, 1184, 693, 16004]
weights = [1 - (x / 41608) for x in num_samples_class]
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)

best_val_loss = float('inf')
patience = 5
patience_counter = 0
correct_predictions = 0
total_predictions = 0

print('Start training...')
for epoch in range(config.num_epochs):
    model.train()
    for i, (texts, labels) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{config.num_epochs}', total=len(train_loader), ncols=100):
        texts = texts.to(device)
        labels = labels.to(device)

        embedded_texts = embedding(texts)

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

            loss = criterion(outputs, labels.long())
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {avg_val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        #torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered')
            break
print('Finished Training')

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
