# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:19:12 2022

@author: mathe
"""

import pandas as pd
from tqdm import tqdm

def performSummary(filteredDf, topic, summaryOnly=False, addData = ["Count", "t1", "t2", "t3", "t4"]):
    bookDictList = []
    summaryDict = {"Topic": topic}
    
    # Add other useful data to the summary df
    rowOneDict = filteredDf.to_dict("records")[0]
    for data in addData:
        summaryDict[data] = rowOneDict[data]
    
    bookList = filteredDf["uri"].drop_duplicates().to_list()
    
    # Calculate main summaries
    booksInTopic = len(bookList)
    msCount = len(filteredDf["ms"].drop_duplicates())
    summaryDict["booksInTopic"] = booksInTopic
    summaryDict["msCount"] = msCount
    
    # Add book specific data
    for book in bookList:
        bookDf = filteredDf[filteredDf["uri"] == book]
        bookCount = len(bookDf["ms"].drop_duplicates())
        summaryDict[book] = bookCount        
        bookDictList.append({"uri": book, "bookCount": bookCount})
    
    
    
    if summaryOnly:        
        return summaryDict
    else:
        booksInTopic = [{"Topic": topic, "booksInTopic" : len(bookList)}]
        msCount = [{"Topic": topic, "msCount": len(filteredDf["ms"].drop_duplicates())}]
        return pd.DataFrame(bookDictList), pd.DataFrame(booksInTopic), pd.DataFrame(msCount), summaryDict

def addSummaryCounts(csvIn, csvOut, csvNoTopicOut, csvTopicSummary):
    
    df = pd.read_csv(csvIn)
    
    ## Fetch all topics
    topicList = df["Topic"].drop_duplicates().to_list()
    
    ## Create empty df and dict into which to concatenate outputs
    outDf = pd.DataFrame()
    summaryDictList = []
    
    ## Loop through topics merging in summary data
    for topic in tqdm(topicList):
        
        ## Check if topic is minus 1 - if so summarise and export separately
        if int(topic) == -1:
            filteredDf = df[df["Topic"] == topic]
            summaryDict = performSummary(filteredDf, topic, summaryOnly=True)
            summaryDictList.append(summaryDict)            
            filteredDf.to_csv(csvNoTopicOut, index=False, encoding='utf-8-sig')
        else:
            filteredDf = df[df["Topic"] == topic]
            bookDictList, booksInTopic, msCount, summaryDict = performSummary(filteredDf, topic)
            
            # For each summary join onto the filteredDf
            filteredDf = filteredDf.merge(bookDictList, on='uri', how='left')
            filteredDf = filteredDf.merge(msCount, on='Topic', how='left')
            filteredDf = filteredDf.merge(booksInTopic, on='Topic', how='left')
            
            # Concatenate onto final output Df and dict
            summaryDictList.append(summaryDict)
            outDf = pd.concat([outDf, filteredDf])
    
    summaryDf = pd.DataFrame(summaryDictList)
    summaryDf.to_csv(csvTopicSummary, index=False, encoding = 'utf-8-sig')
    outDf.to_csv(csvOut, index=False, encoding ='utf-8-sig')

csvIn = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/512-adaptiveSplit-Topics.csv"
csvOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/512-summaries/512-adaptiveSplit-Topics-summarised.csv"
summaryOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/512-summaries/summary-512.csv"
minus1 = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/512-summaries/noTopics.csv"  

addSummaryCounts(csvIn, csvOut, minus1, summaryOut)          