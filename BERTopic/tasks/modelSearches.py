import os
import sys
sys.path.append(os.path.abspath('..'))
from mainFuncs.BERTopic_model import csvToBERTopic


def modelSearches(searchCsv, outCsv, summaryCsv, vizOut, saveModel = None):

    topicModel, summaryDf = csvToBERTopic(searchCsv, outCsv, returnModel=True, sentenceField="phrase", returnSummary=10, returnSummaryCsv = summaryCsv,
    colSummary=[{'col1': 'ms', 'col2': 'uri'}], embeddingModel="CAMeL-Lab/bert-base-arabic-camelbert-ca")

    fig = topicModel.visualize_heirarchy()
    fig.write_html(vizOut)
    
    if saveModel:
        topicModel.save(saveModel)
    return topicModel

if __name__ == "__main__":
    
    inputFile = "../../data/inputs/searchResults/largerWindow/results.csv"
    LargeOutputFolder = "../../data/outputs/searchModelling/"
    outputFolder = "/BERTopic/tasks/output/"
    outCsv = LargeOutputFolder + "results.csv"
    outSummary = outputFolder + "results-summary.csv"
    modelOut = outputFolder + "results.model"
    vizOut = outputFolder + "results-viz.html"

    modelSearches(inputFile, outCsv, outSummary, vizOut, modelOut)



