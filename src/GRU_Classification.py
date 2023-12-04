import torch, string, re
import torch.nn as nn
import config.Config as config
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

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
    def __init__(self, testData):
        self.testData = testData['file_content']
        self.device = self.__getDevice()
        label_encoder = torch.load(config.GRUClassesPath)
        self.class_names = label_encoder.classes_
        self.model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(self.device)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(config.GRUStareDictPath, map_location=self.device))
        self.__classify()

    def __classify(self):
        self.model.eval()
        with torch.no_grad():
            processed_data = self.__getProcessedData(self.testData)
            embedded_data = self.embedding(processed_data)
            predictions = self.model(embedded_data)
            predicted_class_index = torch.argmax(predictions, dim=1).item()
            predicted_class_name = self.class_names[predicted_class_index]
            print('Result:', predicted_class_index, predicted_class_name)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('Using CPU')
            return torch.device('cpu')
        
    def __getProcessedData(self, text):
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        text = word_tokenize(text.lower())
        pattern = re.compile(config.pattern)
        text = [word for word in text if pattern.match(word)]
        text = [word for word in text if word not in config.stopWordList]
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]
        word_to_index = torch.load(config.GRUWordToIndexPath)
        text = [word_to_index.get(word, word_to_index["UNK"]) for word in text]
        if len(text) < config.max_length:
            text += [word_to_index["PAD"]] * (config.max_length - len(text))
        else:
            text = text[:config.max_length]
        return torch.tensor([text], dtype=torch.long).to(self.device)

