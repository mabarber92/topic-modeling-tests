from mainFuncs.InputFromText import InputFromText
from mainFuncs.BERTopic_model import csvToBERTopic
from sentence_transformers import SentenceTransformer, models
from torch import nn
import torch
from multiprocess import Pool
import os
import pandas as pd
from tqdm import tqdm
import time

def modelCorpusOffline(outDir, corpus_base_path=None, metadata_path=None, inCsvDir = None, end_date = 1000, 
                       seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, modelSummary = None, labelCount = 10):
    
        
    if not inCsvDir:
        inCsvDir = outDir
        # Fetch all of the texts and filter
        metadata = pd.read_csv(metadata_path, sep = "\t")
        metadata = metadata[metadata["status"] == "pri"]
        metadata = metadata[metadata["date"] <= end_date]
        metadata["rel_path"] = corpus_base_path + metadata["local_path"].str.split("/master/|\.\./", expand = True, regex=True)[1]
        location_list = metadata["rel_path"].to_list()   
    
        # Loop through text paths and concatenate a df containing all strings split by set parameters
        
        for location in tqdm(location_list):
            if os.path.exists(location):
                uri = ".".join(location.split("/")[-1].split(".")[0:-1])
                print(uri)
                outPath = outDir + "/" + uri + ".csv"
                InputFromText(location, outPath = outPath, seqCap = seqLength, adaptiveSplit=True, shingle=shingle)            
            else:
                print(location + " - path not found")
        
        print("Csv files created")
    else:
        print("Using existing list of csvs")
    
    fullDf = pd.DataFrame()
    for root, dirs, files in os.walk(inCsvDir, topdown=False):
        for name in tqdm(files):
            csvPath = os.path.join(root, name)
            df = pd.read_csv(csvPath)[["ms", "split", "text"]]
            df["uri"] = name.split(".csv")[0]
            fullDf = pd.concat([fullDf, df])
    
    print("Csvs concatenated")
    
    # Pass the full concatenated df to the topic model
    fullDf, topicModel = csvToBERTopic(fullDf, inputType="df",  seqLength = seqLength, embeddingModel = "aubmindlab/bert-base-arabertv02", returnDf=True, returnModel=True)
    
    # Split df into csvs for each uri
    allUris = fullDf["uri"].drop_duplicates.to_list()
    
    for uri in allUris:
        outDest = outDir + "/" + uri + ".csv"
        outDf = fullDf[fullDf["uri"] == uri]
        outDf.drop(columns=["uri"])
        outDf.to_csv(outDest, index=False, encoding='utf-8-sig')
        
    
    # If modelSummary return a csv containing a summary of the model's topics
    if modelSummary:        
        summaryDfCounts = topicModel.get_topic_info()[["Topic", "Count"]]

        topicLabels = topic_model.generate_topic_labels(nr_words = labelCount)
        
        for idx, label in enumerate(topicLabels):
            label = label.split("_")
            topicLabels[idx] = label
        summaryHeading = ["Topic"]
        
        for i in range(0, labelCount):
            summaryHeading.append('t' + str(i))
        summaryDf = pd.DataFrame(topicLabels, columns=summaryHeading)
        summaryDf = summaryDf.merge(summaryDfCounts, on='Topic', how='left')

        summaryDf.to_csv(modelSummary, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    startTime = time.time()
    csvPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus/topicsByText"
    summaryPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpusModelSummary.csv"
    outDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpusModel/"
    modelCorpusOffline(outDir, inCsvDir=csvPath, modelSummary=summaryPath)
    duration = time.time() - startTime
    print("Time to run: " + str(duration))