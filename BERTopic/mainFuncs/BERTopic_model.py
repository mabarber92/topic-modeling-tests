# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:20:21 2022

@author: mathe
"""

import pandas as pd
from bertopic import BERTopic
# from topically import Topically
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import hdbscan
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch import cuda
import torch
import os
from InputFromText import InputFromText
from tqdm import tqdm
from river import stream
from river import cluster
from umap import UMAP

class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self

def check_cuda(model):
	if torch.cuda.is_available():
		cuda_device = 0
		model = model.cuda(cuda_device)
	else:
		cuda_device = -1
	return model,cuda_device

def createModelOnline(dirIn, dirOut=None, transformerModel=None, topicModel=None, n_clusters=500, inputType="csvs", seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, sentenceField="text"):
    
    # Intitate an embedding model if one isn't supplied
    if not transformerModel:
        print("creating embedding model")
        model_name = embeddingModel
        max_seq_length= seqLength
        word_embedding_model = models.Transformer(model_name, max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True)
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                                   out_features=seqLength, 
                                   activation_function=nn.Tanh())
        transformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        transformerModel, cuda = check_cuda(transformerModel)
        print("model loaded")
    
    # Initiate a topic model that is suitable for partial fitting
    if not topicModel:    
        print("Creating the topic model for partial fit")
        umap_model = IncrementalPCA(n_components=5)
        # cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        cluster_model = River(cluster.DBSTREAM())
        vectorizer_model = OnlineCountVectorizer(decay=.01)
        
        topicModel = BERTopic(umap_model=umap_model,                    
                           vectorizer_model=vectorizer_model, hdbscan_model=cluster_model, language="multilingual", verbose=True)
    
    print("commencing Partial fit")
    for root, dirs, files in os.walk(dirIn, topdown=False):
        for name in tqdm(files):
            path = os.path.join(root, name)
            
            # Import the sentences for embedding and modeling - if texts are supplied split them
            if inputType == "texts":
                if shingle:
                    csvName = name + "-" + str(seqLength) + "-shingled-Topics.csv"
                else:
                    csvName = name + "-" + str(seqLength) + "-Topics.csv"
                outPath = os.path.join(dirOut, csvName)
                df = InputFromText(path, outPath, seqCap = seqLength, adaptiveSplit=True, shingle=shingle)
            if inputType == "csvs":
                df = pd.read_csv(path, encoding='utf-8-sig')
            
            # Embed the sentences
            sentences = df[sentenceField].tolist()
            embeds = transformerModel.encode(sentences)
            
            # Partially fit the model using the sentences
            topicModel.partial_fit(df[sentenceField], embeds)
    
    print("Partially fitted Model to Corpus")
    return topicModel
            
            
    
    # If input is a csv move straight to embedding and fitting the model

def csvToBERTopic(csvIn, csvOut=None, inputType = "csv", sentenceField = "text", transformer=None, existingModel = None, returnDf = False, returnModel = False, seqLength = 512, 
                  sortBy = ['t1', 't2', 't3', 't4'], embeddingModel = "aubmindlab/bert-base-arabertv02",
                  topicLimit = None, returnSummary=None, reduceOutliers=None, seed=None, calculateProbabilities = False, existingEmbeds = None, n_neighbors=15):
    """reduceOutliers is populated as a dictionary e.g. {"strategy": "c-tf-idf", "thres": 0.1} or {"strategy": "probabilities", "thres": 0.2}"""
    # Load in input
    if inputType == "csv":
        df = pd.read_csv(csvIn).dropna()
    elif inputType == "df":
        df = csvIn
    
    if existingEmbeds:
        embeds = existingEmbeds
    else:
        # Load embedding model
        print("loading model...")
        if not transformer:
            model_name = embeddingModel
            
            word_embedding_model = models.Transformer(model_name, seqLength)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=True)
            dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                                    out_features=seqLength, 
                                    activation_function=nn.Tanh())
            # It seems we may not need the device set up
            device = "cuda:0" if cuda.is_available() else "cpu"
            print(device)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
            
            print("model loaded")
        else:
            model = transformer
        
        # Take sentence column and pass to encoder
        print("commencing embedding...")
        sentences = df[sentenceField].tolist()
        embeds = model.encode(sentences, show_progress_bar=True)
        print("embeds created...")

    # Set up calculating probs if needed by outlier reduction
    if reduceOutliers:
        if reduceOutliers["strategy"] == "probabilities":
            calculateProbabilities = True
    

    
    # Build topic model - if there is a cap on topics initiate topics using KMeans
    if not existingModel:
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)
        if topicLimit:
            print("Using a topicLimit of: " + str(topicLimit))
            cluster_model = KMeans(n_clusters= topicLimit)
            # REVIST - PASS IN CLUSTER MODEL
            topic_model = BERTopic(calculate_probabilities=calculateProbabilities, language = 'multilingual',umap_model=umap_model)
        
        
        else:
            # If Cuda equals 0, initiate clustering models with GPU capabilities
            # if cuda == 0:
            #     from cuml.cluster import HDBSCAN
            #     from cuml.manifold import UMAP
            #     # Create instances of GPU-accelerated UMAP and HDBSCAN
            #     umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
            #     hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True)

            #     # Pass the above models to be used in BERTopic
            #     topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, language='multilingual')
            # else:
            topic_model = BERTopic(calculate_probabilities=calculateProbabilities, language='multilingual', umap_model=umap_model)
        
        df['Topic'], probabilities = topic_model.fit_transform(df[sentenceField], embeds)
    else:
        topic_model = existingModel
        df['Topic'], probabilities = topic_model.transform(df[sentenceField], embeds)

    print("clustering and topic model built")

    # fit_transform the model to the sentences and embeddings
    # df['Topic'], probabilities = topic_model.fit_transform(df[sentenceField], embeds)
    
    # Add the topic data to the dataframe
    print("fit_transform complete... adding topic data to df")
    
    
    if reduceOutliers:
        print("reducing outliers")
        print(reduceOutliers)
        
        if reduceOutliers["strategy"] == "probabilities":
            df["New Topic"] = topic_model.reduce_outliers(df[sentenceField], df['Topic'], probabilities = probabilities, strategy = "probabilities", threshold=reduceOutliers["thres"])
        else:
            df["New Topic"] = topic_model.reduce_outliers(df[sentenceField], df['Topic'], strategy = reduceOutliers["strategy"], threshold=reduceOutliers["thres"])

        topic_model.update_topics(df[sentenceField], topics=df["New Topic"])
        
    
    topic_info = topic_model.get_topic_info()
    
    df = df.merge(topic_info, left_on='Topic', right_on = "Topic", how='left')
    df[["Topic", 't1', 't2', 't3', 't4']] = df["Name"].str.split("_", expand=True)
    df = df.drop(columns=['Name'])
    df = df.sort_values(by=['t1', 't2', 't3', 't4'])

    if returnSummary:
        summaryDfCounts = topic_model.get_topic_info()[["Topic", "Count"]]
        topicLabels = topic_model.generate_topic_labels(nr_words = returnSummary)

        # Create the headings for a new df
        summaryHeading = ["Topic"]
        
        for i in range(0, returnSummary):
            summaryHeading.append('t' + str(i))
        
        
        summaryDf = pd.DataFrame(topicLabels, columns=["label"])
        summaryDf[summaryHeading] = summaryDf["label"].str.split("_", expand=True)
        summaryDf = summaryDf.drop(columns=['label'])
        summaryDf["Topic"] = summaryDf["Topic"].astype(int)
        print(summaryDf)   
        summaryDf = summaryDf.merge(summaryDfCounts, on='Topic', how='left')
    
    if csvOut:    
        df.to_csv(csvOut, index=False, encoding='utf-8-sig')
    
    # If returnDf is true - return whole df, if returnModel is true, return the model if both are true return both
    # This allows for function to be fed into following processes without reloading the outputs from storage into memory
    if returnDf and returnModel and returnSummary:
        return df, topic_model, summaryDf
    if returnDf and returnSummary:
        return df, summaryDf
    if returnModel and returnSummary:
        return topic_model, summaryDf
    if returnDf and returnModel:
        return df, topic_model
    if returnSummary:
        return summaryDf
    if returnDf:
        return df
    if returnModel:
        return topic_model
    
    
def modelCorpus(inDir, outDir, inputType="csvs", seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, modelSummary = None):
    
    # Intitate a model to loop around and update
    print("creating embedding model")
    model_name = embeddingModel
    max_seq_length= seqLength
    word_embedding_model = models.Transformer(model_name, max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                               out_features=seqLength, 
                               activation_function=nn.Tanh())
    transformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    transformerModel, cuda = check_cuda(transformerModel)
    print("model loaded")
    
    # creatModelOnline will partial_fit() the model by iterating through the corpus and will create the csvs to provide outputs
    topicModel = createModelOnline(inDir, outDir, inputType=inputType, seqLength=seqLength, transformerModel=transformerModel, shingle=shingle)
    
    # Loop through the inDir updating the model with each text and transform
    for root, dirs, files in os.walk(outDir, topdown=False):
        for idx, name in enumerate(files):
            path = os.path.join(root, name)
            print(path)            
            csvToBERTopic(path, path, transformer=transformerModel, existingModel=topicModel, seqLength = seqLength)
                
           
             
    
    # If modelSummary return a csv containing a summary of the model's topics
    if modelSummary:
        summaryDf = topicModel.get_topic_info()
        summaryDf.to_csv(modelSummary, index=False, encoding='utf-8-sig')
    
    return topicModel    




# # Test modelCorpus
# inDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/corpus/MaqriziCorpus"
# outDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/MaqriziPartialFitTest/csvsShingled/"
# model = modelCorpus(inDir, outDir, inputType="texts", shingle=True, modelSummary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/MaqriziPartialFitTest/modelSummaryShingled.csv")


# Test a whole corpus run without online
# corpus_base_path = "D:/OpenITI Corpus/corpus_2022_1_6/"
# metadata_path = "D:/Corpus Stats/2022/OpenITI_metadata_2022-1-6.csv"
# outDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus/"
# modelSummaryDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpusSummary.csv"
# modelCorpusOffline(outDir, inCsvDir = outDir ,
#                    shingle=True, modelSummary = modelSummaryDir, end_date = 500)

# Test new summary option
# inText = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/corpus/MaqriziCorpus/0845Maqrizi.ItticazHunafa.Shamela0000176-ara1.completed"

# seqLength = 512
# df = InputFromText(inText, seqCap = seqLength, adaptiveSplit=True, shingle=True) 
# df, topicSummary = csvToBERTopic(df, inputType="df",  seqLength = seqLength, embeddingModel = "aubmindlab/bert-base-arabertv02", returnDf=True, returnSummary=10)

# """Below is earlier code for subtopic modelling - potentially add ability to pass this into function later """
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

