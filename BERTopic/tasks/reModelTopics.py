import os
import sys
import argparse
sys.path.append(os.path.abspath('..'))
from mainFuncs.BERTopic_model import csvToBERTopic
import pandas as pd
from bertopic import BERTopic


def remodel_topics(topicCsv, outDir, topics, vizOut = True, treeOut = True, saveModel = False, seed=None):
    
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    topicDf = pd.read_csv(topicCsv)
    if type(topics) is not list:
        topics = [topics]
    
    for topic in topics:
        print(topic)        
        filteredDf = topicDf[topicDf["Topic"] == int(topic)]
        filteredDf = filteredDf[["phrase", "uri", "ms"]]
        print(filteredDf.head())
        
        # Create paths for topic csv and summary csv
        outCsv = os.path.join(outDir, "re-modelled-{}.csv".format(topic))
        summaryCsv = os.path.join(outDir, "re-modelled-summary{}.csv".format(topic))

        # Run the model
        topicModel, summaryDf = csvToBERTopic(filteredDf, outCsv, inputType='df', returnModel=True, sentenceField="phrase", returnSummary=10, returnSummaryCsv = summaryCsv,
        colSummary=[{'col1': 'ms', 'col2': 'uri'}], seed=seed, min_topic_size=5)
    
        if treeOut or vizOut:
            print("Getting hierarchical topic labels...")
            hierTopics = topicModel.hierarchical_topics(filteredDf["phrase"])

            if vizOut:
                print("Creating heirarchy viz")
                vizOutPath = os.path.join(outDir, "re-modelled-{}-viz.html".format(topic))                
                fig = topicModel.visualize_hierarchy(hierarchical_topics = hierTopics)
                fig.write_html(vizOutPath)
            
            if treeOut:       
                print("Creating tree")
                treeOutPath = os.path.join(outDir, "re-modelled-{}-tree.txt".format(topic))  
                tree = topicModel.get_topic_tree(hierTopics)
                with open(treeOutPath, "w", encoding='utf-8-sig') as f:
                    f.write(tree)
        
    
            if saveModel:
                print("Saving model")
                modelPath = os.path.join(outDir, "re-modelled-{}.tmodel".format(topic))  
                topicModel.save(modelPath)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', help='Option to save viz - default True', type=bool, default=True)
    parser.add_argument('--tree', help='Option to save tree - default True', type=bool, default=True)
    parser.add_argument('--model', help='Option to save model - default False', type=bool, default=False)
    
    print("Checking args")
    print(str(sys.argv))
    args = parser.parse_args(sys.argv[4:])
    print(args.viz)
    remodel_topics(sys.argv[1], sys.argv[2], sys.argv[3], args.viz, args.tree, args.model)