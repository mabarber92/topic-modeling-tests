import pandas as pd

import os
import sys
sys.path.append(os.path.abspath('..'))
from mainFuncs.BERTopic_model import csvToBERTopic
import pandas as pd
from mainFuncs.addSummaryCounts import performColSummary


path = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/outputs/parameterTesting/embed-models/searchResults/results-arabert.csv"
summaryPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/outputs/parameterTesting/embed-models/searchResults/results-arabert-summary.csv"
df = pd.read_csv(path)
summaryDf = pd.read_csv(summaryPath)

# csvToBERTopic(path, "text-topic-model-output", sentenceField="phrase", n_neighbors=10, colSummary=[{"col1": "ms", "col2": "uri"}],
# reduceOutliers = {"strategy" : "c-tf-idf", "thres": 0.2})

df, summaryDfCounts = performColSummary(df, mainFilter="Topic", col="ms", col2="uri", returnSummary=True)
summaryDf = summaryDf.merge(summaryDfCounts, on="Topic", how='left')
summaryDf.to_csv("test-sum-output.csv", encoding='utf-8-sig')
