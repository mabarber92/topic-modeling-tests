import os
import sys
import argparse
sys.path.append(os.path.abspath('..'))
from mainFuncs.reduceOutliers import reduceOutliers
from utilities.regexCheckTops import regexCheckTops
from bertopic import BERTopic
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compareOutlierThres(topicModel, dataIn, returnDir, compTerms, thresholdStep = 0.2, csvIn = True, graphOut = None, topicModelPath = True):
    if not os.path.exists(returnDir):
        os.mkdir(returnDir)

    if csvIn:
        mainDf = pd.read_csv(dataIn)
    else:
        mainDf = dataIn.copy()
    
    rangeEnd = 1 + 0.1
    divider = 1
    
    if topicModelPath:
        topic_model = BERTopic.load(topicModel)
    else:
        topic_model = topicModel

    while thresholdStep < 1:
        decimalStep = True
        rangeEnd = rangeEnd * 10
        divider = divider * 10
        thresholdStep = thresholdStep * 10

    thresholdStep = int(thresholdStep)
    rangeEnd = int(rangeEnd)
    print(rangeEnd)
    print(thresholdStep)

    resultsDf = pd.DataFrame()

    for i in range(0, rangeEnd, thresholdStep):
        threshold = i/divider
        csvOut = os.path.join(returnDir, "topics-{}-outThres.csv".format(i))
        print("Reducing outliers with {} threshold".format(threshold))

        reducedOutliers = reduceOutliers(topic_model, mainDf, threshold=threshold, csvOut=csvOut, compareOld=True)
        regexResults = regexCheckTops(reducedOutliers, compTerms, csvIn=False)
        
        regexResults["Outlier_thres"] = threshold
        regexResults["Total_terms"] = regexResults[compTerms].sum(axis=1)

        resultsDf = pd.concat([resultsDf, regexResults])
        print("Results added to data frame")
    
    resultsDf["Percent_of_count"] = resultsDf["Total_terms"]/resultsDf["Count"]*100

    # Remove topics where there is no change and log those separately
    topicList = resultsDf["Topic"].drop_duplicates().to_list()
    noChangeDf = pd.DataFrame()
    noCountChangeDf = pd.DataFrame()
    for topic in topicList:
        filteredResults = resultsDf[resultsDf["Topic"] == topic]
        uniqueValues = filteredResults[compTerms].drop_duplicates()
        if len(uniqueValues) == 1:
            resultsDf = resultsDf[resultsDf["Topic"] != topic]
            noChangeDf = pd.concat([noChangeDf, filteredResults])
        
        uniqueCount = filteredResults["Count"].drop_duplicates()
        if len(uniqueCount) == 1:
            noCountChangeDf = pd.concat([noCountChangeDf, filteredResults])
    
    unchangedCount = len(noChangeDf["Topic"].drop_duplicates().to_list())
    unchangedLength = len(noChangeDf["Topic"].drop_duplicates().to_list())
    print("{} Topics are unchanged according to search parameters".format(unchangedCount))
    print("{} Topics have not changed length across selected thresholds".format(unchangedLength))

    noChangePath = os.path.join(returnDir, "unchanged-topics.csv")
    noChangeDf.to_csv(noChangePath, index=False, encoding='utf-8-sig')

    noCountChangePath = os.path.join(returnDir, "unchanged-counts.csv")
    noCountChangeDf.to_csv(noCountChangePath, index=False, encoding="utf-8-sig")

  
    csvOutPath = os.path.join(returnDir, "changed-topics.csv")
    resultsDf.to_csv(csvOutPath, index=False, encoding='utf-8-sig')
    
    if graphOut:
        print("Creating line charts from results")
        resultsDf["Topic"] = resultsDf["Topic"].astype('category')        
        g = sns.lineplot(data=resultsDf, x="Outlier_thres", y="Total_terms", hue="Topic")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        figure = g.get_figure()
        figure.savefig(graphOut + "-absoluteterms.png", dpi=300, bbox_inches='tight')
        plt.clf()
        
        g = sns.lineplot(data=resultsDf, x="Outlier_thres", y="Percent_of_count", hue="Topic")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        figure = g.get_figure()
        figure.savefig(graphOut + "-perc.png", dpi=300, bbox_inches='tight')
        plt.clf()

        g = sns.lineplot(data=resultsDf, x="Outlier_thres", y="Count", hue="Topic")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        figure = g.get_figure()
        figure.savefig(graphOut + "-topic-count.png", dpi=300, bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', help='Choose the length of each step for comparing thresholds - default 0.1', type=int, default=0.1)
    
    parser.add_argument('--graphOut', help='Option to graph the data output - provide a path for saving the graph', type=str, default=None)
    parser.add_argument('--termsCsv', help='Supply a csv listing regex terms to be used - put regex under column "Term"', type=str)
    
    
    args = parser.parse_args(sys.argv[4:])
    terms = pd.read_csv(args.termsCsv)["Term"].to_list()
    print(terms)

    compareOutlierThres(sys.argv[1], sys.argv[2], sys.argv[3], terms, graphOut=args.graphOut, thresholdStep=args.steps)
    
