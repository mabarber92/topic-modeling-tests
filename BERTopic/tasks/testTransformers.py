from mainFuncs.BERTopic_model import csvToBERTopic

if __name__ == '__main__':
    testCsv = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/largerWindow/results.csv"
    Out = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-arabert.csv"
    Summary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-arabert-summary.csv"
    SummaryDf = csvToBERTopic(testCsv, Out, sentenceField="phrase", returnSummary=10, seed=10)
    SummaryDf.to_csv(Summary, encoding="utf-8-sig")

    testCsv = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/largerWindow/results.csv"
    Out = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-CAMeLBERT-CA.csv"
    Summary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-CAMeLBERT-CA-summary.csv"
    SummaryDf = csvToBERTopic(testCsv, Out, sentenceField="phrase", returnSummary=10, embeddingModel = "CAMeL-Lab/bert-base-arabic-camelbert-ca", seed=10)
    SummaryDf.to_csv(Summary, encoding="utf-8-sig")

    testCsv = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/largerWindow/results.csv"
    Out = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-camelbert-mix-pos-msa.csv"
    Summary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/embed-models/searchResults/results-camelbert-mix-pos-msa-summary.csv"
    SummaryDf = csvToBERTopic(testCsv, Out, sentenceField="phrase", returnSummary=10, embeddingModel = "CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa", seed=10)
    SummaryDf.to_csv(Summary, encoding="utf-8-sig")

    # for i in range(0,10):
    #     i = i/10
    #     print(i)
    #     LargeWindowOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/outlierTesting/c-tfidf-thres-seed-10/results-arabert-outthres-" + str(i) + ".csv"
    #     LargeSummary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/outlierTesting/c-tfidf-thres-seed-10/results-arabert-outthres-" + str(i) + "-summary.csv"
    #     SummaryDf = csvToBERTopic(csvLargeWindow, LargeWindowOut, sentenceField="text", returnSummary=10, reduceOutliers=i, seed=10)
    #     SummaryDf.to_csv(LargeSummary, encoding="utf-8-sig")

    
    # csvSmallWindow = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results.csv"
    # SmallWindowOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results-arabert.csv"
    # SmallSummary = ":/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results-arabert-summary.csv"
    # SummaryDf = csvToBERTopic(csvSmallWindow, SmallWindowOut, sentenceField="phrase", returnSummary=10, reduceOutliers=True)
    # SummaryDf.to_csv(SmallSummary)

    # csvLargeWindow = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/largerWindow/results.csv"
    # LargeWindowOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/results-camelbert.csv"
    # LargeSummary = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/results-camelbert-summary.csv"
    # SummaryDf = csvToBERTopic(csvLargeWindow, LargeWindowOut, sentenceField="phrase", returnSummary=10, reduceOutliers=True, embeddingModel = "CAMeL-Lab/bert-base-arabic-camelbert-ca")
    # SummaryDf.to_csv(LargeSummary)
    
    # csvSmallWindow = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results.csv"
    # SmallWindowOut = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results-camelbert.csv"
    # SmallSummary = ":/Users/mathe/Documents/Github-repos/topic-modeling-tests/searchResults/smallWindow/all_results-camelbert-summary.csv"
    # SummaryDf = csvToBERTopic(csvSmallWindow, SmallWindowOut, sentenceField="phrase", returnSummary=10, reduceOutliers=True, embeddingModel = "CAMeL-Lab/bert-base-arabic-camelbert-ca")
    # SummaryDf.to_csv(SmallSummary)