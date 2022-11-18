# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:20:21 2022

@author: mathe
"""

import pandas as pd
from bertopic import BERTopic
# from topically import Topically
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, models
from torch import nn
import torch

def check_cuda(model):
	if torch.cuda.is_available():
		cuda_device = 0
		model = model.cuda(cuda_device)
	else:
		cuda_device = -1
	return model,cuda_device


df = pd.read_csv("C:/Users/mathe/Documents/Github-repos/fitna-study/search_phrases/Yusuf_search/all_results.csv")

print("commencing embedding...")
model_name = "aubmindlab/bert-base-arabertv02"
max_seq_length=256

word_embedding_model = models.Transformer(model_name, max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True)
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                           out_features=256, 
                           activation_function=nn.Tanh())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
model, cuda = check_cuda(model)
print("model loaded")
print(cuda)
sentences = df["phrase"].tolist()

embeds = model.encode(sentences)
print("embeds created...")

main_tops = 300

# Load and initialize BERTopic to use KMeans clustering with 8 clusters only.
# cluster_model = KMeans(n_clusters= main_tops)
topic_model = BERTopic(language='multilingual')
print("clustering and topic model built")

# df is a dataframe. df['title'] is the column of text we're modeling
df['topic'], probabilities = topic_model.fit_transform(df['phrase'], embeds)

# # Add step to perform subtopic classification for larger sentence collections
# sub_tops = 50

# df_out = pd.DataFrame()
# sub_cluster_model = KMeans(n_clusters= sub_tops)
# sub_topic_model = BERTopic(hdbscan_model=sub_cluster_model, language='multilingual')

# for i in range(0, main_tops):
#     print("creating subtopics for topic: " + str(i))
#     topic = df[df['topic'] == i]
#     sub_embeds = model.encode(topic['phrase'].tolist())    
#     topic['subtopic'], probabilities = sub_topic_model.fit_transform(topic['phrase'], sub_embeds)
#     df_out = pd.concat([df_out, topic])
    


# print("topic field created... passing to app")
# # Load topically
# app = Topically('ETjfG97N4TgmVwq7v3MZzYUPSbZ2wQKHrXE2K0Be')

# # name clusters
# df['topic_name'] = app.name_topics((df['phrase'], df['topic']))[0]
df = df.sort_values(by=['topic'])

df.to_csv("Yusuf_search/topic_model_test_arabertv02_noset2.csv", index=False, encoding='utf-8-sig')