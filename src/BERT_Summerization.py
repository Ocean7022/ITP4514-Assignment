from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans

class BERT_Summerization:
    def __init__(self, text):
        sentences = text['file_content'].split('. ')
        device = self.__getDevice()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.__summerization(sentences, tokenizer, model)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'\nUsing GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('\nUsing CPU')
            return torch.device('cpu')

    def __summerization(self, sentences, tokenizer, model):
        # Tokenize each sentence using BERT and create attention masks
        encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Disable gradient computation, perform forward pass, and get outputs
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the output of the [CLS] token of each sentence as the sentence embedding
            sentence_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()

        # Use KMeans for clustering analysis, set the number of clusters to be formed
        kmeans = KMeans(n_clusters=3, n_init=10)
        # Perform clustering on sentence embeddings
        kmeans.fit(sentence_embeddings)

        # For each cluster center, find the closest sentence
        closest_sentences_indices = []
        for cluster_center in kmeans.cluster_centers_:
            # Calculate the distance of each sentence to the cluster center
            distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
            # Find the index of the closest sentence
            closest_sentence_index = distances.argmin().item()
            closest_sentences_indices.append(closest_sentence_index)

        # Ensure the selected sentences are unique
        unique_sentence_indices = list(set(closest_sentences_indices))

        # If the number of selected sentences is less than 3, choose additional closest sentences
        while len(unique_sentence_indices) < 3 and len(unique_sentence_indices) < len(sentences):
            for i, cluster_center in enumerate(kmeans.cluster_centers_):
                if len(unique_sentence_indices) == 3:
                    break
                distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
                for index in distances.argsort():
                    if index.item() not in unique_sentence_indices:
                        unique_sentence_indices.append(index.item())
                        break

        # Extract summary sentences based on indices
        summary_sentences = [sentences[idx] for idx in unique_sentence_indices if idx < len(sentences)]

        # Combine summary sentences into a summary
        summary = '. '.join(summary_sentences)
        print('\nSummary:\n  ', summary + ".")
