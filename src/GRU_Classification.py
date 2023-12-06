import string, re
import torch.nn as nn
import config.Config as config
import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from nltk.stem import PorterStemmer
import torch.nn.functional as F
from GRU_DataPorcess import GRU_DataProcess as DP

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
        text = text['file_content']
        device = self.__getDevice()
        self.classify(device, text)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('Using CPU')
            return torch.device('cpu')

    def classify(self, device, text):
        model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)
        model.load_state_dict(torch.load(config.GRUClassificationModelPath))
        model.eval()

        word_to_index = torch.load(config.GRUWordToIndexPath)
        classes = torch.load(config.GRUClassesPath)
        label_encoder = torch.load(config.GRULabelEncoderPath)
        embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)

        porcessed_text = DP.process_text(text)
        indexed_text = [word_to_index.get(word, word_to_index["UNK"]) for word in porcessed_text][:config.max_length]
        indexed_tensor = torch.tensor([indexed_text], dtype=torch.long)
        padded_texts = pad_sequence(indexed_tensor, batch_first=True, padding_value=word_to_index["PAD"]).to(device)

        output = model(embedding(padded_texts))
        _, predicted = torch.max(output, dim=1)

        # Print results
        print(label_encoder.inverse_transform([predicted.item()])[0])



