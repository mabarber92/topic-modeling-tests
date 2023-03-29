from bertopic import BERTopic
import pandas as pd

def reduceOutliers(topicModel, dataIn, threshold, csvOut = None, sentenceField= "phrase", modelAsPath = False, csvIn = False, compareOld = False):
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
        
    
    if csvOut:
        df.to_csv(csvOut, index=False, encoding='utf-8-sig')

    return df