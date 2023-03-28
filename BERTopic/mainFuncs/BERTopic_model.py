# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:20:21 2022

@author: mathe
"""

import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import KMeans
from tqdm import tqdm
from umap import UMAP
from mainFuncs.addSummaryCounts import performColSummary
from mainFuncs.initialiseEmbedModel import initialiseEmbedModel
from numpy import random

def csvToBERTopic(csvIn, csvOut=None, inputType = "csv", sentenceField = "text", transformer=None, existingModel = None, returnDf = False, returnModel = False, seqLength = 512, 
                  sortBy = ['t1', 't2', 't3', 't4'], embeddingModel = "CAMeL-Lab/bert-base-arabic-camelbert-ca",
                  topicLimit = None, returnSummary=None, reduceOutliers=None, seed=None, calculateProbabilities = False, existingEmbeds = None, n_neighbors=15, colSummary = None,
                  returnSummaryCsv = None, min_topic_size = 10):
    """reduceOutliers is populated as a dictionary e.g. {"strategy": "c-tf-idf", "thres": 0.1} or {"strategy": "probabilities", "thres": 0.2}
    colSummary should be formatted as list of dicts for each col or pairs of col to summarise on: e.g. [{"col1": "ms", "col2": "uri"}]"""
    if seed:
        seed = random.RandomState(seed=seed)
        print(seed)
    # Load in input
    if inputType == "csv":
        df = pd.read_csv(csvIn).dropna()
    elif inputType == "df":
        df = csvIn
    
    # If no embeddings are passed to the function create embeddings using specified model
    if existingEmbeds:
        embeds = existingEmbeds
    else:
        # Load embedding model if it has not already been passed to the model        
        if not transformer:
            model = initialiseEmbedModel(embeddingModel, seqLength)
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
    # If an existing topic model has been passed, transform the new sentences and embeddings without initialising a new model
    if not existingModel:
        print("Creating new model and fitting and transforming")
        umapModel = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=seed)
        if topicLimit:
            print("Using a topicLimit of: " + str(topicLimit))
            cluster_model = KMeans(n_clusters= topicLimit)
            # REVIST - PASS IN CLUSTER MODEL
            topic_model = BERTopic(calculate_probabilities=calculateProbabilities, language = 'multilingual',umap_model=umapModel, min_topic_size = min_topic_size)
        
        
        else:
            print("Creating topic model without limit")
            topic_model = BERTopic(calculate_probabilities=calculateProbabilities, language='multilingual', umap_model=umapModel, min_topic_size = min_topic_size)
            print(topic_model.umap_model)
        df['Topic'], probabilities = topic_model.fit_transform(df[sentenceField], embeds)
        


    else:
        print("Transforming using existing model")
        topic_model = existingModel
        df['Topic'], probabilities = topic_model.transform(df[sentenceField], embeds)
    
    # Add the topic data to the dataframe
    print("fit_transform complete... adding topic data to df")
    
    # If an outlier reduction method is being adopted then apply it according to set parameters and merge the updated topic info
    # Otherise merge the topic info without updating it
    # !! Replace this with the specific reduceOutliers function !!
    if reduceOutliers:
        print("reducing outliers")
        print(reduceOutliers)
        
        if reduceOutliers["strategy"] == "probabilities":
            df["New Topic"] = topic_model.reduce_outliers(df[sentenceField], df['Topic'], probabilities = probabilities, strategy = "probabilities", threshold=reduceOutliers["thres"])
        else:
            df["New Topic"] = topic_model.reduce_outliers(df[sentenceField], df['Topic'], strategy = reduceOutliers["strategy"], threshold=reduceOutliers["thres"])

        topic_model.update_topics(df[sentenceField], topics=df["New Topic"])     
        topic_info = topic_model.get_topic_info()
        df = df.merge(topic_info, left_on='New Topic', right_on = "Topic", how='left', suffixes=("Old", "New"))
        df = df.drop(columns=["New Topic"])
    else:
        topic_info = topic_model.get_topic_info()
        df = df.merge(topic_info, left_on='Topic', right_on = "Topic", how='left')
    
    print("Adding labels")
    # Split the topic labels off and then sort the dataframe according to chosen approach
    df[["Topic", 't1', 't2', 't3', 't4']] = df["Name"].str.split("_", expand=True)
    df = df.drop(columns=['Name'])
    df = df.sort_values(by=sortBy)

    # If there are columns specified for summarisation - summarise on them
    if colSummary:
        for sumCrit in colSummary:            
            if "col1" in sumCrit.keys() and "col2" in sumCrit.keys():
                df, colSummaryDf = performColSummary(df, sumCrit["col1"], mainFilter="Topic", col2 = sumCrit["col2"], returnSummary=True)
            elif "col1" in sumCrit.keys():
                df, colSummaryDf = performColSummary(df, sumCrit["col1"], mainFilter="Topic", returnSummary=True)

        print(colSummaryDf)
        print(type(colSummaryDf["Topic"].to_list()[0]))
        colSummaryDf["Topic"] = colSummaryDf["Topic"].astype('int32')
        print(type(colSummaryDf["Topic"].to_list()[0]))
    
    # If a summary data frame is needed, create it
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
        summaryDf["Topic"] = summaryDf["Topic"].astype('int32')
        print(summaryDf)   
        summaryDf = summaryDf.merge(summaryDfCounts, on='Topic', how='left')

        if colSummary:
            summaryDf = summaryDf.merge(colSummaryDf, on="Topic", how='left')
        if returnSummaryCsv:
            summaryDf.to_csv(returnSummaryCsv, index=False, encoding='utf-8-sig')
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
