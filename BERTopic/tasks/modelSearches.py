import os
import sys
sys.path.append(os.path.abspath('..'))
from mainFuncs.BERTopic_model import csvToBERTopic
import pandas as pd
from bertopic import BERTopic


def modelSearches(searchCsv, outCsv, summaryCsv, vizOut, treeOut, saveModel = None, seed=None, existingModel=None):
    searchDf = pd.read_csv(searchCsv)
    print(seed)
    if not existingModel:
        
        topicModel, summaryDf = csvToBERTopic(searchDf, outCsv, inputType='df', returnModel=True, sentenceField="phrase", returnSummary=10, returnSummaryCsv = summaryCsv,
        colSummary=[{'col1': 'ms', 'col2': 'uri'}], embeddingModel="CAMeL-Lab/bert-base-arabic-camelbert-ca", seed=seed)
    else:
        topicModel = BERTopic.load(existingModel)
    
        print("Getting hierarchical topic labels...")
    hierTopics = topicModel.hierarchical_topics(searchDf["phrase"])

    print("Creating heirarchy viz")
    fig = topicModel.visualize_hierarchy(hierarchical_topics = hierTopics)
    fig.write_html(vizOut)

    print("Creating tree")
    tree = topicModel.get_topic_tree(hierTopics)
    with open(treeOut, "w", encoding='utf-8-sig') as f:
        f.write(tree)
        
    
    if saveModel:
        topicModel.save(saveModel)
    return topicModel

if __name__ == "__main__":
    
    inputFile = "../../data/inputs/searchResults/largerWindow/results.csv"
    LargeOutputFolder = "../../data/outputs/searchModelling/"
    outputFolder = "./output/"
    outCsv = LargeOutputFolder + "results-camelbert-seed100-run1.csv"
    outSummary = outputFolder + "results-camelbert-seed100-summary-run1.csv"
    modelOut = outputFolder + "results-camelbert-seed100.model"
    vizOut = outputFolder + "results-camelbert-seed100-viz-whier-run1.html"
    treeOut = outputFolder + "results-camelbert-seed100-tree-whier-run1.txt"


    # cutInput = outputFolder + "cutInput.csv"
    # dfCut = pd.read_csv(inputFile).loc[0:5000]
    # dfCut.to_csv(cutInput, index = False, encoding = 'utf-8-sig')

    modelSearches(inputFile, outCsv, outSummary, vizOut, treeOut, modelOut, seed=100)




