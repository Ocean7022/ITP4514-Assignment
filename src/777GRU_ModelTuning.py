import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import config.Config as config
import nltk, string, os, re
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import torch.nn.functional as F

class NewsDataset(Dataset):
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
        self.classes = []
        if os.path.exists(config.GRUProcessedDataSetPath):
            select = input('\nDo you want to delete old dataSet? (y/n): ')
            if select == 'y':
                os.remove(config.GRUProcessedDataSetPath)
            elif select == 'n':
                pass
            else:
                print('Invalid input, please try again.')
                self.__init__()

        self.train_dataset, self.test_dataset = self.__dataPorcess()
        self.train_dataset, self.test_dataset = self.__saveClasses(self.train_dataset, self.test_dataset)

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)

        self.device = self.__getDevice()
        self.model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(self.device)
        self.class_weights = self.__countClassWeights(self.train_dataset).to(self.device)    
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights).to(self.device)
        #criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(self.device)
        torch.save(self.model.state_dict(), config.GRULabelEncoderPath)
        self.__startTraining()
        self.__startTesting()
    
    def __startTraining(self):
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        correct_predictions = 0
        total_predictions = 0
        print('Start training...')
        for epoch in range(config.num_epochs):
            self.model.train()
            for i, (texts, labels) in tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch+1}/{config.num_epochs}', total=len(self.train_loader), ncols=100):
                texts = texts.to(self.device)
                labels = labels.to(self.device)

                embedded_texts = self.embedding(texts)
                #embedded_texts = [torch.mean(embedded_text, dim=0) for embedded_text in embedded_texts]
                #embedded_texts = [embedded_text.unsqueeze(1) for embedded_text in embedded_texts]
                
                outputs = self.model(embedded_texts)
                #outputs = self.model(torch.stack(embedded_texts))
                loss = self.criterion(outputs, labels.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for texts, labels in self.test_loader:
                    texts = texts.to(self.device)
                    labels = labels.to(self.device)

                    embedded_texts = self.embedding(texts)
                    #embedded_texts = [torch.mean(embedded_text, dim=0) for embedded_text in embedded_texts]
                    #embedded_texts = [embedded_text.unsqueeze(1) for embedded_text in embedded_texts]
                    
                    outputs = self.model(embedded_texts)
                    #outputs = self.model(torch.stack(embedded_texts))
                    loss = self.criterion(outputs, labels.long())
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(self.test_loader)
            val_accuracy = 100 * correct_predictions / total_predictions
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {avg_val_loss:.4f}, Test Accuracy: {val_accuracy:.2f}%')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), config.GRUClassificationModelPath)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    break
        print('Finished Training')

    def __startTesting(self):
        print('Start testing...')
        with torch.no_grad():
            correct = 0
            total = 0
            index = 0
            for texts, labels in self.test_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                embedded_texts = self.embedding(texts)
                #embedded_texts = [torch.mean(embedded_text, dim=0) for embedded_text in embedded_texts]
                #embedded_texts = [embedded_text.unsqueeze(1) for embedded_text in embedded_texts]
                
                outputs = self.model(embedded_texts)
                #outputs = self.model(torch.stack(embedded_texts))
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for idx in range(predicted.size(0)):
                    print('\nResult for Test Sample #{}:'.format(idx))
                    print('Predicted Label:', self.classes[predicted[idx].item()])
                    print('Actual Label:', self.classes[labels[idx].item()])
                    print('Probabilities:')
                    for i, class_name in enumerate(self.classes):
                        print(f"  {class_name}: {probabilities[idx][i].item():.4f}")

            print('Test Accuracy: {} %'.format(100 * correct / total))
        print('Finished Testing\n')

    def __take500ItemsInEachType(self, texts, labels):
        new_texts = []
        new_labels = []
        label_counts = Counter(labels)
        for label in label_counts.keys():
            count = 0
            for i in range(len(labels)):
                if labels[i] == label:
                    new_texts.append(texts[i])
                    new_labels.append(labels[i])
                    count += 1
                    if count == 500:
                        break
        print('New DataSet Size:', len(texts))
        return new_texts, new_labels

    def __dataPorcess(self):
        if os.path.exists(config.GRUProcessedDataSetPath):
            print('Processed DataSet already exists.')
            return self.__getTrainAndTestLoder(torch.load(config.GRUProcessedDataSetPath))
        else:
            print('Processed DataSet does not exist, start processing...')
            print('Reading DataSet...')
            with open(config.dataSetPath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print('DataSet read successfully.', len(data), 'items in total.')

            nltk.download('punkt')

            texts = [item['title'] + '. ' + item['content'] for item in data]
            labels = [item['category'] for item in data]
            texts, labels = self.__cleanToShortData(texts, labels)
            #texts, labels = self.__take500ItemsInEachType(texts, labels)
           # label_encoder = LabelEncoder()
           # labels = label_encoder.fit_transform(labels)
           # print(torch.tensor(labels))

            translator = str.maketrans('', '', string.punctuation)
            texts = [text.translate(translator) for text in tqdm(texts, desc='Removing Punctuations', ncols=100)]
            texts = [word_tokenize(text.lower()) for text in tqdm(texts, desc='Tokenizing', ncols=100)]
            pattern = re.compile(config.pattern)
            texts = [[word for word in text if pattern.match(word)] for text in tqdm(texts, desc='Removing Non-English Words', ncols=100)]
            stemmer = PorterStemmer()
            texts = [[stemmer.stem(word) for word in text] for text in tqdm(texts, desc='Stemming', ncols=100)]
            texts = [[word for word in text if word not in config.stopWordList] for text in tqdm(texts, desc='Removing Stopwords', ncols=100)]

            #self.__countAvgLength(texts)

            word_freq = Counter(word for sentence in tqdm(texts, desc='Processing Texts', ncols=100) for word in sentence)
            # customizing the vocabulary size
            vocab = [word for word, freq in tqdm(word_freq.most_common(config.vocab_size - 2), desc='Creating Vocabulary', ncols=100)]
            # full size vocabulary
            #vocab = [word for word, freq in tqdm(word_freq.most_common(len(word_freq) - 2), desc='Creating Vocabulary', ncols=100)]
            #config.vocab_size = len(vocab)
            vocab.append("UNK")
            vocab.append("PAD")
            print('Vocabulary Size:', len(vocab))

            word_to_index = {word: index for index, word in enumerate(vocab)}
            torch.save(word_to_index, config.GRUWordToIndexPath)
            texts = [[word_to_index.get(word, word_to_index["UNK"]) for word in text] for text in tqdm(texts, desc='Converting to Sequences', ncols=100)]
            #print(texts[3])
            #exit()

            padded_sequences = []
            for seq in tqdm(texts, desc='Padding Sequences', ncols=100):
                if len(seq) < config.max_length:
                    seq += [word_to_index["PAD"]] * (config.max_length - len(seq))
                else:
                    seq = seq[:config.max_length]
                padded_sequences.append(torch.tensor(seq, dtype=torch.long))

            padded_sequences = torch.stack(padded_sequences, dim=0)

            
            dataset = NewsDataset(padded_sequences, labels)
            torch.save(dataset, config.GRUProcessedDataSetPath)
            return self.__getTrainAndTestLoder(dataset)

    def __cleanToShortData(self, texts, labels, minSentenceLength = 10):
        newTexts = []
        newLabels = []
        for label, text in tqdm(zip(labels, texts), desc='Removing too short data', ncols=100):
            if len(sent_tokenize(text)) > minSentenceLength:
                newTexts.append(text)
                newLabels.append(label)
        print('Removed', len(texts) - len(newTexts), 'too short data')
        print('New DataSet Size:', len(newTexts))
        print('New Labels Size:', len(newLabels))
        return newTexts, newLabels

    def __getTrainAndTestLoder(self, dataset):
        total_size = len(dataset)
        train_size = int(total_size * config.train_ratio)
        test_size = total_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('Using CPU')
            return torch.device('cpu')

    def __countAvgLength(texts):
        text_lengths = [len(text) for text in texts]
        sortedText = sorted(text_lengths)
        #print('Total Texts:', sortedText)
        print('Top 10 Text Lengths:', sortedText[-10:])
        print('Bottom 10 Text Lengths:', sortedText[:10])
        print('Avg:', int(sum(text_lengths) / len(text_lengths)))
        print('75%:', int(np.percentile(text_lengths, 75)))
        print('90%:', int(np.percentile(text_lengths, 90)))
        print('95%:', int(np.percentile(text_lengths, 95)))

        sns.set(style='whitegrid')
        plt.figure(figsize=(12, 6))
        sns.histplot(text_lengths, bins=50, kde=True)

        plt.title('Text Length Distribution')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.savefig('../img/GUR-Tuning/textsLengthDistribution.png')

    def __saveClasses(self, train_dataset, test_dataset):
        train_texts = [train_dataset[i][0] for i in range(len(train_dataset))]
        test_texts = [test_dataset[i][0] for i in range(len(test_dataset))]
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]
        train_label_encoder = LabelEncoder()
        test_label_encoder = LabelEncoder()
        train_labels = train_label_encoder.fit_transform(train_labels)
        test_labels = test_label_encoder.fit_transform(test_labels)
        self.classes = train_label_encoder.classes_
        torch.save(self.classes, config.GRUClassWeightPath)
        return NewsDataset(train_texts, torch.tensor(train_labels)), NewsDataset(test_texts, torch.tensor(test_labels))

    def __countClassWeights(self, dataset):
        labels = [dataset[i][1].item() for i in range(len(dataset))]
        label_counts = Counter(labels)
        num_samples_class = [label_counts[i] for i in range(config.num_classes)]
        weights = [1 - (x / sum(num_samples_class)) for x in num_samples_class]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        print('Weights:', normalized_weights)
        #for i in range(config.num_classes):
        #    print(f'{self.classes[i]}: {normalized_weights[i]:.4f}')
        return torch.tensor(normalized_weights, dtype=torch.float)
