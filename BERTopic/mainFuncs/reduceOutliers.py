from bertopic import BERTopic
import pandas as pd
from mainFuncs.addSummaryCounts import performColSummary

def reduceOutliers(topicModel, dataIn, threshold, csvOut = None, sentenceField= "phrase", modelAsPath = False, csvIn = False, compareOld = False, colSummary=[{"col1": "ms", "col2":"uri"}]):
    # Load model
    if modelAsPath:
        topic_model = BERTopic.load(topicModel)
    else:
        topic_model = topicModel

    # Load data
    if csvIn:
        df = pd.read_csv(dataIn)
    else:
        df = dataIn.copy()

    print("Data and model loaded...")

    df["New Topic"] = topic_model.reduce_outliers(df[sentenceField], df['Topic'], strategy = 'c-tf-idf', threshold=threshold)

    print("Outliers reduced... updating topics")

    topic_model.update_topics(df[sentenceField], topics=df["New Topic"])     
    topic_info = topic_model.get_topic_info()
    df = df.merge(topic_info, left_on='New Topic', right_on = "Topic", how='left', suffixes=("Old", "New"))
    df = df.drop(columns=["New Topic", "Name"])

    print("New topics merged...")
    
    if not compareOld:
        print("Dropping old topic assignment")
        df = df.drop(columns = ["TopicOld"])
        df = df.drop(columns = ["CountOld"])
        
    df = df.rename(columns = {"TopicNew": "Topic"})
    df = df.rename(columns = {"CountNew": "Count"})
        
    # If there are columns specified for summarisation - summarise on them
    if colSummary:
        print("Summarising columns on {}".format(str(colSummary)))
        for sumCrit in colSummary:            
            if "col1" in sumCrit.keys() and "col2" in sumCrit.keys():
                df, colSummaryDf = performColSummary(df, sumCrit["col1"], mainFilter="Topic", col2 = sumCrit["col2"], returnSummary=True)
            elif "col1" in sumCrit.keys():
                df, colSummaryDf = performColSummary(df, sumCrit["col1"], mainFilter="Topic", returnSummary=True)

        
        colSummaryDf["Topic"] = colSummaryDf["Topic"].astype('int32')

    if csvOut:
        df.to_csv(csvOut, index=False, encoding='utf-8-sig')
 
    return df