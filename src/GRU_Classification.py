import string, re
import torch.nn as nn
import config.Config as config
import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from nltk.stem import PorterStemmer
import torch.nn.functional as F

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

class GRU_Classification:
    def __init__(self, text):
        self.text = text['file_content']
        self.device = self.__getDevice()
        self.classes = torch.load(config.GRUClassesPath)
        print(self.classes)
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].replace('_', ' ')
        print(self.classes)
        self.model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(config.GRUStareDictPath, map_location=self.device))
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(self.device)
        self.word_to_index = torch.load(config.GRUWordToIndexPath)
        self.classify()

    def classify(self):
        processed_text = self.__process_text(self.text)
        print(processed_text)
        embedded_text = self.embedding(processed_text)
        print(embedded_text.shape)

        # Prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(embedded_text)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            predicted_class_index = predicted.item()
            predicted_class_name = self.classes[predicted_class_index]

        # Print results
        print('Result:', predicted_class_index, predicted_class_name)

        # Print probabilities for each class
        for i, class_name in enumerate(self.classes):
            print(f"{class_name}: {probabilities[0][i].item():.4f}")

    def __process_text(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = word_tokenize(text.lower())
        pattern = re.compile(config.pattern)
        text = [word for word in text if pattern.match(word)]
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if word not in config.stopWordList]
        
        text = [self.word_to_index.get(word, self.word_to_index["UNK"]) for word in text]
        #print(text)
        if len(text) < config.max_length:
            text += [self.word_to_index["PAD"]] * (config.max_length - len(text))
        else:
            text = text[:config.max_length]
        text_tensor = torch.tensor([text], dtype=torch.long)
        #print(text_tensor)
        #print(len(text))
        #exit()
        return text_tensor.to(self.device)

    def __getDevice(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
