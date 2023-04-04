import sys
import pandas as pd

def mergeOutlierCountsSummary(summaryCsv, fullTopicCsv, labelExt="WithOutliers", topicCol = "Topic"):
    summaryDf = pd.read_csv(summaryCsv)
    fullTopicDf = pd.read_csv(fullTopicCsv)[[topicCol, "Count", "msCount", "uriCount"]].drop_duplicates()
    summaryDf = summaryDf.merge(fullTopicDf, left_on='Topic', right_on = "Topic", how='left', suffixes=("", labelExt))
    summaryDf.to_csv(summaryCsv, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    if len(sys.argv) == 4:
        mergeOutlierCountsSummary(sys.argv[1], sys.argv[2], sys.argv[3], topicCol = "Topic")
    else:
        mergeOutlierCountsSummary(sys.argv[1], sys.argv[2], topicCol = "Topic")
