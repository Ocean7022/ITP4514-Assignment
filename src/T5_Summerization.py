from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class T5_Summerization:
    def __init__(self, text):
        device = self.__getDevice()
        model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
        tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
        summary = self.generate_summary(text['file_content'], model, device, tokenizer)
        print('\nSummart:\n  ', summary)

    def __getDevice(self):
        if torch.cuda.is_available():
            print(f'\nUsing GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
            return torch.device('cuda')
        else:
            print('\nUsing CPU')
            return torch.device('cpu')    

    def generate_summary(self, text, model, device, tokenizer, max_length=512, num_beams=4, length_penalty=1.0, min_length=50, max_output_length=150):
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
        summary_ids = model.generate(inputs, num_beams=num_beams, length_penalty=length_penalty, min_length=min_length, max_length=max_output_length, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
