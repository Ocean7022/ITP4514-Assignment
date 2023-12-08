import torch.nn as nn
import config.Config as config
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
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
    def __init__(self):
        self.device = self.__getDevice()
        self.model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(config.GRUClassificationModelPath, map_location=self.device))
        self.model.eval()
        self.word_to_index = torch.load(config.GRUWordToIndexPath, map_location=self.device)
        self.label_encoder = torch.load(config.GRULabelEncoderPath, map_location=self.device)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(self.device)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'\nUsing GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('\nUsing CPU')
            return torch.device('cpu')

    def classify(self, text):
        processed_text = DP.process_text(text['file_content'])
        indexed_text = [self.word_to_index.get(word, self.word_to_index["UNK"]) for word in processed_text][:config.max_length]
        indexed_tensor = torch.tensor([indexed_text], dtype=torch.long)
        padded_texts = pad_sequence(indexed_tensor, batch_first=True, padding_value=self.word_to_index["PAD"]).to(self.device)

        result = []
        for i in range(10):
            output = self.model(self.embedding(padded_texts))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
            predicted_class = self.label_encoder.inverse_transform([predicted.item()])[0]
            result.append(predicted_class)

        count = Counter(result)
        most_common_result, count = count.most_common(1)[0]
        print(f"\n - Predicted class: [ {most_common_result} ]")
        for idx, prob in enumerate(probabilities[0]):
            class_name = self.label_encoder.inverse_transform([idx])[0]
            print(f"      {class_name}: {prob.item():.4f}")



