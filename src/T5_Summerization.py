from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch
import config.Config as config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} '.format(device))

model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

def generate_summary(text, max_length=512, num_beams=4, length_penalty=1.0, min_length=30, max_output_length=130):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
    summary_ids = model.generate(inputs, num_beams=num_beams, length_penalty=length_penalty, min_length=min_length, max_length=max_output_length, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

df = pd.read_json(config.dataSetPath)
df["text"] = df["title"] + ". " + df["content"]
text_s = df['text'][3]

summary = generate_summary(text_s)
print('\n\n',summary)
