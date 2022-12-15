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

def csvToBERTopic(csvIn, csvOut, sentenceField = "text", returnDf = False, returnModel = False, seqLength = 512, 
                  sortBy = ['t1', 't2', 't3', 't4'], embeddingModel = "aubmindlab/bert-base-arabertv02",
                  topicLimit = None):
    
    # Load in input
    df = pd.read_csv(csvIn).dropna()
    
    # Load embedding model
    print("loading model...")
    model_name = embeddingModel
    max_seq_length= seqLength
    word_embedding_model = models.Transformer(model_name, max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                               out_features=seqLength, 
                               activation_function=nn.Tanh())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    print("model loaded")
    print("checking cuda...")
    model, cuda = check_cuda(model)
    print(cuda)
    
    # Take sentence column and pass to encoder
    print("commencing embedding...")
    sentences = df[sentenceField].tolist()
    embeds = model.encode(sentences)
    print("embeds created...")
    
    # Build topic model - if there is a cap on topics initiate topics using KMeans
    if topicLimit:
        print("Using a topicLimit of: " + str(topicLimit))
        cluster_model = KMeans(n_clusters= topicLimit)
        # REVIST - PASS IN CLUSTER MODEL
        topic_model = BERTopic(language = 'multilingual')
    
    else:
        topic_model = BERTopic(language='multilingual')

    print("clustering and topic model built")

    # fit_transform the model to the sentences and embeddings
    df['Topic'], probabilities = topic_model.fit_transform(df[sentenceField], embeds)
    
    # Add the topic data to the dataframe
    print("fit_transform complete... adding topic data to df")
    
    topic_info = topic_model.get_topic_info()
    df = df.merge(topic_info, on='Topic', how='left')
    df[["Topic", 't1', 't2', 't3', 't4']] = df["Name"].str.split("_", expand=True)
    df = df.drop(columns=['Name'])
    df = df.sort_values(by=['t1', 't2', 't3', 't4'])

    df.to_csv(csvOut, index=False, encoding='utf-8-sig')
    
    # If returnDf is true - return whole df, if returnModel is true, return the model if both are true return both
    # This allows for function to be fed into following processes without reloading the outputs from storage into memory
    if returnDf:
        return df
    if returnModel:
        return topic_model
    if returnDf and returnModel:
        return df, topic_model
    


"""Below is earlier code for subtopic modelling - potentially add ability to pass this into function later """
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

