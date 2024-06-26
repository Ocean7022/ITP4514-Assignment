import torch
import torch.nn as nn
import torch.optim as optim
import json, nltk, random
import config.Config as config
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter
from GRU_DataPorcess import GRU_DataProcess as DP
from torch.nn.utils.rnn import pad_sequence

class TensorDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRU_ModelTuning:
    def __init__(self):
        nltk.download('punkt')
        device = self.__getDevice()
        texts, labels = self.__loadData()
        label_encoder, encoded_labels, class_weight = self.__encodeLabelsAndCountClassWeight(labels)
        processed_texts = self._getProcessedTexts(texts)
        word_to_index, indexed_texts = self.__indexTexts(processed_texts)
        val_loader = self._takeValLoader(texts, labels)
        train_loader = self.__getTrainLoader(indexed_texts, encoded_labels)
        trained_model = self.__train(train_loader, val_loader, class_weight, device)
        self.__save(trained_model, label_encoder, word_to_index, class_weight)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('Using CPU')
            return torch.device('cpu')
        
    def __loadData(self):
        with open(config.dataSetPath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print('DataSet read successfully.', len(data), 'items in total.')

        texts = [item['title'] + '. ' + item['content'] for item in data]
        labels = [item['category'] for item in data]
        return texts, labels
    
    def _takeValLoader(self, texts, labels):
        classes = list(set(labels))

        new_texts = []
        new_labels = []
        for class_name in classes:
            count = 0
            while count < 100:
                random_index = random.randint(0, len(labels) - 1)
                if labels[random_index] == class_name:
                    new_texts.append(texts[random_index])
                    new_labels.append(labels[random_index])
                    del texts[random_index]
                    del labels[random_index]
                    count += 1

        label_encoder, encoded_labels, class_weight = self.__encodeLabelsAndCountClassWeight(new_labels)
        processed_texts = [
            DP.process_text(text) for text in new_texts
        ]
        word_to_index, indexed_texts = self.__indexTexts(processed_texts)
        val_loader = self.__getTrainLoader(indexed_texts, encoded_labels)
        return val_loader
    
    def __encodeLabelsAndCountClassWeight(self, labels):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        label_counts = Counter(encoded_labels)
        total_counts = sum(label_counts.values())
        class_weight = {class_index: total_counts / label_counts[class_index] for class_index in label_counts}

        for class_index in class_weight:
            class_weight[class_index] = class_weight[class_index] / max(class_weight.values())
            
        for class_index, weight in class_weight.items():
            class_name = label_encoder.inverse_transform([class_index])[0]
            print(f"Class: {class_name}, Weight: {weight:.4f}")

        return label_encoder, encoded_labels, class_weight

    def _getProcessedTexts(self, texts):
        processed_texts = [
            DP.process_text(text) for text in tqdm(texts, desc='Processing Texts', ncols=100)
        ]
        return processed_texts

    def __indexTexts(self, texts):
        vocab = Counter(word for text in tqdm(texts, desc='Creating Vocabulary', ncols=100) for word in text)
        most_common_vocab = vocab.most_common(config.vocab_size - 2)
        print('Vocab size:', len(most_common_vocab))

        word_to_index = {"UNK": 0, "PAD": 1}
        for i, (word, _) in enumerate(most_common_vocab, start=2):
            word_to_index[word] = i
        
        indexed_texts = []
        for text in texts:
            indexed_text = [word_to_index.get(word, word_to_index["UNK"]) for word in text][:config.max_length]
            indexed_texts.append(torch.tensor(indexed_text, dtype=torch.long))

        padded_texts = pad_sequence(indexed_texts, batch_first=True, padding_value=word_to_index["PAD"])
        return word_to_index, padded_texts

    def __getTrainLoader(self, texts, labels):
        text_tensors = texts.clone().detach()
        label_tensors = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(text_tensors, label_tensors)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        return train_loader
    
    def __train(self, train_loader, val_loader, class_weight, device):
        model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(class_weight.values()), dtype=torch.float)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)

        patience = 10
        best_loss = float('inf')
        patience_counter = 0
        early_stop = False

        for epoch in range(config.num_epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            # Training Loop
            model.train()
            for i, (texts, labels) in tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{config.num_epochs}', total=len(train_loader), ncols=100):
                texts = texts.to(device)
                labels = labels.to(device)

                embedded_texts = embedding(texts)

                optimizer.zero_grad()
                outputs = model(embedded_texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            avg_loss = total_loss / len(train_loader)
            accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.4f}%")

            # Testing Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for texts, labels in val_loader:
                    texts = texts.to(device)
                    labels = labels.to(device)
                    embedded_texts = embedding(texts)
                    outputs = model(embedded_texts)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                trained_model = model # save the model
            else:
                patience_counter += 1
                if patience_counter == patience:
                    print("Early stopping!")
                    early_stop = True
                    break
                
        if not early_stop:
            print("Training completed without early stopping.")

        return trained_model

    def __save(self, model, label_encoder, word_to_index, class_weight):
        torch.save(model.state_dict(), config.GRUClassificationModelPath)
        torch.save(label_encoder, config.GRULabelEncoderPath)
        torch.save(word_to_index, config.GRUWordToIndexPath)
        torch.save(class_weight, config.GRUClassWeightPath)

