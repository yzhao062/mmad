import pandas as pd
import numpy as np
import gensim.downloader as api
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download("punkt")

wv = api.load('word2vec-google-news-300')
data = pd.read_excel("/Users/wenw/Desktop/multimodal/train_cleaned.xlsx")
sentences = data["tweet_text"].values




stop_words = set(stopwords.words('english'))

tokenized_sent = []
for s in sentences:
    token = word_tokenize(s.lower())
    filtered_sentence = [w for w in token if not w in stop_words] 
    new_words= [word for word in filtered_sentence if word.isalnum()]
    tokenized_sent.append(new_words)

word2vec_emebedings = []
for sen in tokenized_sent:
    tmp = []
    for item in sen:
        try:
            vec = wv[item]
            tmp.append(vec)
        except KeyError:
            continue
    word2vec_emebedings.append(np.array(tmp).mean(axis= 0))



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embedding(sequence):
    input_ids = torch.tensor(tokenizer.encode(sequence, add_special_tokens=True, max_length=512, truncation=True)).unsqueeze(0)
    return model(input_ids)[1].data.numpy()

bert_cls_embeding = []
count = 0 
for sen in sentences:
    bert_cls_embeding.append(get_embedding(sen).squeeze())
    count+=1
    if count%500 ==0:
        print(count)


np.save("/Users/xiliu/Desktop/multimodal/bert.npy",bert_cls_embeding)
np.save("/Users/xiliu/Desktop/multimodal/word2vec.npy",word2vec_emebedings)


