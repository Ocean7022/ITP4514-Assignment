from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.cluster import KMeans

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#df = pd.read_json("./data/dataset.json")
#df["text"] = df["title"] + ". " + df["content"]

text_s = "The $250 million civil fraud case against Donald Trump came to a close on Monday in the same manner it began – with the former president waging a martyrdom campaign as he testified under oath and the judge repeatedly admonishing him and his defense team for refusing to answer questions directly. Trump, at times seemingly unable to stop himself from melting into a temper tantrum, claimed repeatedly, “It's election interference.” “We have a very hostile judge,” he said at one point. “I don’t have a jury. I wanted a jury.” To be clear, this particular type of case proceeding – determining damages in a civil fraud case – doesn’t use juries. Perhaps the most fiery exchange occurred just before lunch break, when the lead prosecutor, Kevin Wallace, asked Trump whether he agreed with the attorney general's position that “the statement of financial conditions were overstated.” Instead of answering the question directly, Trump said, “I think she is a political hack. She used this case to try to become governor. … This is a disgrace. All you have to do is read the legal papers and the scholars. This is a political witch hunt, and I think she should be ashamed of herself.” Trump then turned his anger on Engoron, saying that the judge ruled against me before he knew anything about me. He called me a fraud, and he didn’t know anything about me. ... The fraud is on the court – not on me.” At the start of the day, Engoron seemed to ask Trump’s defense team for help keeping him on task and answering questions directly instead of grandstanding and providing long-winded answers that failed to address the specific question. But Trump’s two lead attorneys, Chris Kise and Alina Habba, did little to help. In fact, Kise, who referred to Trump as the “former and future chief executive of the United States,” said it would be in the best of interest of the judge to hear what Trump had to say – an unusual legal tactic for a bench trial in which the judge decides the penalties. Habba, for her part, tore into Engoron, claiming that he “already predetermined that my client committed fraud before we even walked into this courtroom.” Engoron did find, based on court filings, that Trump and his adult sons were liable for fraud and canceled the Trump Organization’s business certificates, siding with New York Attorney General Letitia James, who accused the former president and his business associates of inflating the value of his net worth and properties on financial statements used to secure loans. James is seeking $250 million in compensatory damages along with a ban on the Trumps serving as officers of a business in New York and on the company from engaging in business transactions for five years. While Trump faces 91 criminal charges stemming from four separate indictments – two of which charge him with undermining democracy by attempting to overturn his 2020 presidential election loss – it’s the civil lawsuit from James that seems to have unnerved Trump the most. Since the trial began in late September, Trump has been fined twice by Engoron for breaking a gag order that bars him from making public comments about the judge’s staff, he has stormed out of the courtroom and been made to watch as two of his sons, Donald Trump Jr. and Eric Trump, testified last week. The proceedings have also prompted him to post animated musings on his social media site, TruthSocial, where he maintains the legitimacy of his business dealings and claims that he’s a victim of political persecution. Despite the contentious atmosphere, the attorney general’s prosecutorial team was able to elicit some answers from Trump about how his assets were valued, in which he said that he considered some of his properties as valued too high and others as valued too low – based largely on whether or not they included an estimate of his brand value – but that he didn’t make a big deal about it or ask for changes. That narrative is a markedly different one than Trump’s two adult sons provided in testimony last week. They said that they were unfamiliar with how those numbers were calculated and simply relied on others who had much more expertise in that area, and then signed off on the documents as prepared. When asked about various financial statements, Trump repeatedly attempted to use as a defense the so-called “worthless statement” clause, which disclaims responsibility for the calculations and insists that anyone using it must do their own financial analysis. But Engoron already dismissed that defense in finding Trump liable for fraud. In his ruling, Engoron wrote that the “worthless statement” clause “does not rise to the level of an enforceable disclaimer and cannot be used to insulate fraud as to facts peculiarly within defendants’ knowledge, even vis-a-vis sophisticated recipients.” Ivanka Trump, his daughter, is set to testify on Wednesday, after which the attorney general’s prosecuting team says they will provide closing arguments before handing it over to Trump’s defense team to present their own case. Kise said he expects to begin next week, with plans to wrap up by Dec. 15. The defense team is expected to call many of the witnesses who have already testified, including Trump himself and his two adult sons. In a curious exchange at the end of the day, Habba referenced plans to make a motion for a mistrial or directed verdict later this week in relation to the gag order – and more specifically, to the judge’s law clerk. Engoron noted that doing so may trigger the gag order itself but said he’d allow it through a process known as “order to show cause.” The process allows the judge to see the motion but prevents the motion from being filed until the judge signs off on it. Addressing reporters outside the courtroom at the end of the day, Trump repeated claims that the case is “a scam” “This is a case that should have never been brought, and it’s a case that should now be dismissed,” he said. “Everything we did was absolutely right.” James also chatted with reporters after court was adjourned for the day, saying that she expected him to “ramble and hurl insults,” was not at all surprised by the strategy and that she will not be bullied or harassed. “The numbers don't lie. And Mr. Trump obviously can engage in all of these distractions, and that is exactly what he did – what he committed on the stand today, engaging in distractions and engaging in name-calling,” she said."
text = text_s.replace('“', '').replace('”', '')

sentences = text.split('. ')

# 对每个句子进行编码，并创建attention mask
encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

is_cuda = torch.cuda.is_available()
print("Using GPU:", is_cuda)
if torch.cuda.is_available():
    model = model.to('cuda')
    input_ids = input_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')

# 使用BERT模型和attention_mask
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    # 取每个句子的[CLS]标记的输出作为句子的嵌入表示
    sentence_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()

# 使用KMeans聚类来确定哪些句子最重要
kmeans = KMeans(n_clusters=3, n_init=10)  # 设置n_clusters为5
kmeans.fit(sentence_embeddings)

# 为每个聚类中心找到最接近的句子
closest_sentences_indices = []
for cluster_center in kmeans.cluster_centers_:
    distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
    closest_sentence_index = distances.argmin().item()
    closest_sentences_indices.append(closest_sentence_index)

# 确保选择的句子是唯一的
unique_sentence_indices = list(set(closest_sentences_indices))

# If there are less than five selected sentences, select the remaining closest sentences
while len(unique_sentence_indices) < 3 and len(unique_sentence_indices) < len(sentences):
    for i, cluster_center in enumerate(kmeans.cluster_centers_):
        if len(unique_sentence_indices) == 3:
            break
        distances = torch.norm(torch.tensor(sentence_embeddings) - torch.tensor(cluster_center), dim=1)
        for index in distances.argsort():
            if index.item() not in unique_sentence_indices:
                unique_sentence_indices.append(index.item())
                break

# 提取句子
summary_sentences = [sentences[idx] for idx in unique_sentence_indices if idx < len(sentences)]

# 组合句子
summary = '. '.join(summary_sentences)
print(summary + ".")
