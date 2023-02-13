# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:37:19 2022

@author: mathe
"""

import pandas as pd
import seaborn as sns
from tqdm import tqdm

def TopicsPerMs(csvIn, dropFreq=60):
    
    # Load in data
    df = pd.read_csv(csvIn)[["uri", "ms", "Topic"]]
    
    # Fetch list of books
    bookList = df["uri"].drop_duplicates().to_list()
    
    outDictList = []
    #Loop through books and create MS List
    for book in tqdm(bookList):
        print(book)
        filteredDf = df[df["uri"] == book]
        msList = filteredDf["ms"].drop_duplicates().to_list()
        
        droppedTopics = []
        totalMs = len(msList)
        topicList = filteredDf["Topic"].drop_duplicates().to_list()
        for topic in topicList:
            msTopicCount = len(filteredDf[filteredDf["Topic"] == topic]["ms"].drop_duplicates())
            if msTopicCount/totalMs*100 > dropFreq:
                print(topic)
                print(totalMs)
                print(msTopicCount)
                
                print(msTopicCount/totalMs*100)
                droppedTopics.append(topic)
        print(droppedTopics)
        filteredDf = filteredDf[~filteredDf['Topic'].isin(droppedTopics)]
        
        
        # Loop through MS List and create DF with counts
        for ms in msList:
            msDf = filteredDf[filteredDf["ms"] == ms]
            # if len(msDf) == 0:
            #     print(ms)
            #     print("No topics")
            
            msTopicCount = len(msDf["Topic"].drop_duplicates())
            outDictList.append({"ms": ms, "uri": book, "topicCount": msTopicCount, "seqCount" : len(msDf)})
    
    dfOut = pd.DataFrame(outDictList)
    dfOut = dfOut.sort_values(by=["uri", "ms"])
    
    return dfOut
    
def graphTopicsPerMs(csvsIn, savePath):
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    # Pass csvs to the TopicsPerMs and concatenate output, storing filenames
    dataDf = pd.DataFrame()
    
    if type(csvsIn) is not list:
        csvsIn = [csvsIn]
        
    for csvIn in csvsIn:
        df = TopicsPerMs(csvIn)
        df["file"] = csvIn.split("/")[-1]
        dataDf = pd.concat([dataDf, df])
    
    # Graph in Facet grid using rows for each book and filenames as hue
    g = sns.relplot(data=dataDf, x="ms", y="topicCount", hue="file", row="uri", facet_kws=dict(sharex=False), alpha=0.3, size="seqCount")
    
    g.fig.subplots_adjust(top=1.4)
    fig = g.fig
    fig.savefig(savePath, dpi=300, bbox_inches='tight')

csvsIn = ["C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/256-summaries/256-adaptiveSplit-Topics-summarised.csv",
          "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/512-summaries/512-adaptiveSplit-Topics-summarised.csv"]

savePath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/msTopicCounts.png"

graphTopicsPerMs(csvsIn, savePath)