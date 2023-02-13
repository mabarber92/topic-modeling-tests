# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:41:47 2022

@author: mathe
"""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
from math import ceil
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def annotate(data, **kws):
    n = len(data)
    ax = plt.gca()
    ax.text(.1, .6, f"Milestones = {n}", transform=ax.transAxes)



def plotTopicDist(data, title, savePath, df = False, topicFilter = None, plot='scatter', uriComp = False, sharedy = True):
    if not df:
        df = pd.read_csv(data)
    else:
        df = data.copy()
    df = df[df["Topic"] != -1]
    if topicFilter:
        df = df[df['Topic'].isin(topicFilter)]
        
    else:
        topicFilter = range(df['Topic'].min(), df['Topic'].max()+1)
    if plot == 'scatter':
        sns.scatterplot(data=df, x="ms", y="Topic", hue="split", alpha=0.3)
    if plot == 'hist':
        cols = ['Topic', 't1', 't2', 't3', 't4']
        df["Title"] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if uriComp:
            
            df[["author", "book"]] = df["uri"].str.split(".", expand=True).iloc[:,[0,1]]
            df["authorBook"] = df["author"].str.cat(df[["book"]], sep = ".")
            
            g = sns.FacetGrid(data=df, row="Topic", col="authorBook", sharey=sharedy)
            g.map(sns.histplot, "ms", binwidth=100)
            # title = get_display(arabic_reshaper.reshape("العربي"))
            axes = g.axes.flatten()
            
            axLoc = 0
            
            for i in topicFilter:
                TopicWords = df[df["Topic"] == i][["Topic", "t1", "t2", "t3", "t4", "Count"]].values.tolist()[0]
                TopicWords[0] = str(TopicWords[0])
                TopicWords[-1] = "\nTotal Size: " + str(TopicWords[-1])
                
                TopicWords = ", ".join(TopicWords)
                newTitle = get_display(arabic_reshaper.reshape(TopicWords))
                g.axes[axLoc,0].set_ylabel(newTitle, rotation='horizontal', horizontalalignment ='right')
                axLoc = axLoc + 1
                print(".", end="")
                
                # g.map_dataframe(annotate)
            g.set_titles(template="{col_name}" )
            for (row_val, col_val), ax in g.axes_dict.items():
                filteredDf = df[df["Topic"] == row_val]
                filteredDf = filteredDf[filteredDf["authorBook"] == col_val]["ms"]
                splitLen = len(filteredDf)
                msLen = len(filteredDf.drop_duplicates())
                newTitle = str(col_val) + ",\n MS Splits: " + str(splitLen) + " MS Unique: " + str(msLen)
                ax.set_title(newTitle)                
                if splitLen == 0:
                    ax.set_facecolor("#001b17")
                else:
                    ax.set_facecolor(".95")
                print(".", end="")
            
        else:
            g = sns.FacetGrid(data=df, col="Topic", col_wrap=5, sharey=sharedy)
            g.map(sns.histplot, "ms", binwidth=100)
            # title = get_display(arabic_reshaper.reshape("العربي"))
            axes = g.axes.flatten()
            for i in range(df['Topic'].min(), df['Topic'].max()+1):
                TopicWords = df[df["Topic"] == i][["Topic", "t1", "t2", "t3", "t4", "Count"]].values.tolist()[0]
                TopicWords[0] = str(TopicWords[0])
                TopicWords[-1] = "\nSize: " + str(TopicWords[-1])
                print(TopicWords)
                TopicWords = ", ".join(TopicWords)
                newTitle = get_display(arabic_reshaper.reshape(TopicWords))
                axes[i].set_title(newTitle)
        g.fig.subplots_adjust(top=1.4)
        fig = g.fig
        fig.savefig(savePath, dpi=300, bbox_inches='tight')
        # g.fig.suptitle(title)
        # g.set_titles(col_template=title)
        
def bookFocussedComp(csvIn, outDir, sharedy=False, graphMax = 25):
    df = pd.read_csv(csvIn)
    df = df[df["Topic"] != -1]
    bookList = df["uri"].drop_duplicates().tolist()
    print(bookList)
    print("\n------------------------\n")
    for book in bookList:
        print(book)
        print("\n------------------------\n")
        imageDir = os.path.join(outDir, book)
        if not os.path.exists(imageDir):
            os.mkdir(imageDir)
        bookDf = df[df["uri"] == book]
        topics = bookDf["Topic"].sort_values().drop_duplicates().tolist()        
        topicCount = len(topics)        
        if len(topics) > graphMax:            
            for i in tqdm(range(0, topicCount, graphMax)):
                topicFocus = topics[i:i+25]
                
                plotTitle = imageDir + "/" + book + "-" + str(min(topicFocus)) + "-" + str(max(topicFocus)) + ".png"
                plotTopicDist(df, "", plotTitle, df=True, topicFilter = topicFocus, plot='hist', 
                              uriComp = True, sharedy = sharedy)
        else:
            plotTitle = imageDir + "/" + book + str(min(topics)) + str(max(topics)) + ".png"
            plotTopicDist(df, "", plotTitle, df=True, topicFilter = topics, plot='hist', 
                          uriComp = True, sharedy = sharedy)
        print("\n------------------------\n")



csvPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/256-adaptiveSplit-Topics.csv"
outDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/graphs/256-split/"
bookFocussedComp(csvPath, outDir) 


# topics = []
# for i in range(0,51):
#     topics.append(i)
# plotTopicDist(csvPath, "", "Maqrizicorpus-top-topics-256-shingled-0-50.png", topicFilter = topics, plot='hist', uriComp = True, sharedy = False)

# topics=[]
# for i in range(50,101):
#     topics.append(i)
# plotTopicDist(csvPath, "", "Maqrizicorpus-top-topics-256-shingled-50-100.png", topicFilter = topics, plot='hist', uriComp = True, sharedy = False)

## SAMPLE CODE FOR IMPROVING AXIS LABELS
# for (row_val, col_val), ax in g.axes_dict.items():
#     if row_val == "Lunch" and col_val == "Female":
#         ax.set_facecolor(".95")
#     else:
#         ax.set_facecolor((0, 0, 0, 0))