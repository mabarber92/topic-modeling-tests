import pandas as pd


from BERTopic_model import csvToBERTopic
import pandas as pd

path = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/inputs/searchResults/largerWindow/results.csv"
df = pd.read_csv(path)

csvToBERTopic(path, "text-topic-model-output", sentenceField="phrase", n_neighbors=10, colSummary=[{"col1": "ms", "col2": "uri"}],
reduceOutliers = {"strategy" : "c-tf-idf", "thres": 0.2})

# df, summaryDf = colSummary(df, mainFilter="Topic", col="ms", col2="uri", returnSummary=True)
# print(df.head().transpose())
# summaryDf.to_csv("test-sum-output.csv", encoding='utf-8-sig')
