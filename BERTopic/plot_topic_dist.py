# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:41:47 2022

@author: mathe
"""

import seaborn as sns
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display

def plotTopicDist(csv, title, savePath, topicFilter = None, plot='scatter'):
    df = pd.read_csv(csv)
    df = df[df["Topic"] != -1]
    if topicFilter:
        df = df[df['Topic'].isin(topicFilter)]
    if plot == 'scatter':
        sns.scatterplot(data=df, x="ms", y="Topic", hue="split", alpha=0.3)
    if plot == 'hist':
        cols = ['Topic', 't1', 't2', 't3', 't4']
        df["Title"] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        g = sns.FacetGrid(data=df, col="Topic", col_wrap=5)
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
        

csvPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/Maqrizi.Mawaciz-seq-512-adaptiveSplit-Topics.csv"
topics = []
for i in range(0,101):
    topics.append(i)
plotTopicDist(csvPath, "", "Maqrizi-Mawaciz-top-topics.png", topicFilter = topics, plot='hist')