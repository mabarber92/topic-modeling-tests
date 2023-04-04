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
import arabic_reshaper
from bidi.algorithm import get_display

def compareOutlierThres(topicModel, dataIn, returnDir, compTermsCsv, thresholdStep = 0.2, csvIn = True, graphOut = True, topicModelPath = True, graphZeros=True):
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

    compTermsDf = pd.read_csv(compTermsCsv)
    compTerms = compTermsDf["Term"].to_list()

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

    csvOutPath = os.path.join(returnDir, "changed-topics.csv")
    resultsDf.to_csv(csvOutPath, index=False, encoding='utf-8-sig')

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
    
    if len(noCountChangeDf) > 0 or len(noChangeDf) > 0:
        unchangedCount = len(noCountChangeDf["Topic"].drop_duplicates().to_list())
        unchangedLength = len(noChangeDf["Topic"].drop_duplicates().to_list())
        print("{} Topics are unchanged according to search parameters".format(unchangedCount))
        print("{} Topics have not changed length across selected thresholds".format(unchangedLength))

        noChangePath = os.path.join(returnDir, "unchanged-topics.csv")
        noChangeDf.to_csv(noChangePath, index=False, encoding='utf-8-sig')

        noCountChangePath = os.path.join(returnDir, "unchanged-counts.csv")
        noCountChangeDf.to_csv(noCountChangePath, index=False, encoding="utf-8-sig")

  
    
    
    if graphOut:
        print("Creating line charts from results")

        sns.set(rc={'figure.figsize':(11.7,8.27)})

        # Add an output directory for the graphs
        graphDir = os.path.join(returnDir, "graphs/")
        if not os.path.exists(graphDir):
            os.mkdir(graphDir)
        
        # Loop through the term groups, produce a graph for each term group
        for termGroup in compTermsDf["Group"].to_list():
            graphPath = os.path.join(graphDir, str(termGroup))
            termsList = compTermsDf[compTermsDf["Group"] == termGroup]["Term"].to_list()
            
            # Create df matching topics to terms - for comparing in figure
            termCompDf = pd.DataFrame()

            for term in termsList:
                termFilteredTopicsDf = resultsDf[resultsDf[term] > 0]
                termFilteredTopics = termFilteredTopicsDf["Topic"].drop_duplicates().to_list()
                termFiltered = resultsDf[resultsDf["Topic"].isin(termFilteredTopics)]
                termFiltered["Term"] = term
                termFiltered["_count"] = termFiltered[term]
                termFiltered["Type"] = "Term Count"
                termCompDf = pd.concat([termCompDf, termFiltered])

                termFiltered["_count"] = termFiltered["Count"]
                termFiltered["Type"] = "Topic Length"
                termCompDf = pd.concat([termCompDf, termFiltered])

                for outThres in termFilteredTopicsDf["Outlier_thres"].drop_duplicates().to_list():
                    
                    thresFiltered = termFilteredTopicsDf[termFilteredTopicsDf["Outlier_thres"] == outThres]
                    
                    summaryRow = thresFiltered.iloc[[0]]
                    summaryRow["Term"] = term
                    summaryRow["Topic"] = "Total Topics"
                    summaryRow["_count"] = len(thresFiltered["Topic"].drop_duplicates())
                    summaryRow["Type"] = "Topic Count"
                    print(summaryRow)
                    termCompDf = pd.concat([termCompDf, summaryRow])

        
         
            
            

            
            
            # 
            #     termFiltered["Term"] = "No Terms Found"
                
            #     termFiltered["_count"] = termFiltered["Total_terms"]
            #     termFiltered["Type"] = "Term Count"
            #     termCompDf = pd.concat([termCompDf, termFiltered])

            #     termFiltered["_count"] = termFiltered["Count"]
            #     termFiltered["Type"] = "Topic Length"
            #     termCompDf = pd.concat([termCompDf, termFiltered])
            
            termCompDf["Topic"] = termCompDf["Topic"].astype('category')    

            no_terms = len(compTerms)
            if no_terms > 5:
                col_wrap = 5
            else:
                col_wrap = no_terms
            g = sns.FacetGrid(data=termCompDf, row="Term", col="Type", sharey=False, sharex=True)
            g.map(sns.lineplot, "Outlier_thres", "_count", "Topic")        
            g.add_legend()
            
            for (row_val, col_val), ax in g.axes_dict.items():
                print(ax)
                print(get_display(arabic_reshaper.reshape(row_val)))
                ax.set_title(col_val)
                if col_val == "Term Count":
                    new_label = get_display(arabic_reshaper.reshape(row_val))
                    ax.set_ylabel(new_label, rotation='horizontal', horizontalalignment ='right')
                else:
                    ax.set_ylabel("", rotation='horizontal', horizontalalignment ='right')


            figure = g.fig       
            figure.savefig(graphPath + "-terms-compared.png", dpi=300, bbox_inches='tight')
            plt.clf()

            # Remove rows that contain Topic of "Total Topics" - as messy for these graphs
            termCompDf["Topic"] = termCompDf["Topic"].astype('str') 
            termCompDf = termCompDf[termCompDf["Topic"] != "Total Topics"]
            termCompDf["Topic"] = termCompDf["Topic"].astype('category') 

                
            g = sns.lineplot(data=termCompDf, x="Outlier_thres", y="Total_terms", hue="Topic")
            sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            figure = g.get_figure()
            figure.savefig(graphPath + "-absoluteterms.png", dpi=300, bbox_inches='tight')
            plt.clf()
            
            g = sns.lineplot(data=termCompDf, x="Outlier_thres", y="Percent_of_count", hue="Topic")
            sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            figure = g.get_figure()
            figure.savefig(graphPath + "-perc.png", dpi=300, bbox_inches='tight')
            plt.clf()

        
        # Create a topic count graph for topics that have no terms identified
        if graphZeros:
            

            termFiltered = resultsDf[resultsDf["Total_terms"] == 0]
            termFiltered["Topic"] = termFiltered["Topic"].astype('category') 
            g = sns.lineplot(data=termFiltered, x="Outlier_thres", y="Count", hue="Topic")
            sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            figure = g.get_figure()
            figure.savefig(graphPath + "-topic-no-term-count.png", dpi=300, bbox_inches='tight')
            plt.clf()
            
            unchangedTops = noChangeDf["Topic"].drop_duplicates().to_list()
            changedTopsDf = resultsDf[~resultsDf["Topic"].isin(unchangedTops)]
            changedTopsDf["Topic"] = changedTopsDf["Topic"].astype('category') 
            g = sns.lineplot(data=changedTopsDf, x="Outlier_thres", y="Count", hue="Topic")
            sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            figure = g.get_figure()
            figure.savefig(graphPath + "-topic-count-all-changed.png", dpi=300, bbox_inches='tight')
            plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', help='Choose the length of each step for comparing thresholds - default 0.1', type=int, default=0.1)
    parser.add_argument('--graphOut', help='Option to graph the data output - provide a path for saving the graph', type=bool, default=True)
    parser.add_argument('--termsCsv', help='Supply a csv listing regex terms to be used - put regex under column "Term"', type=str)
    
    
    args = parser.parse_args(sys.argv[4:])
    terms = pd.read_csv(args.termsCsv)["Term"].to_list()
    print(terms)

    compareOutlierThres(sys.argv[1], sys.argv[2], sys.argv[3], compTermsCsv = args.termsCsv, graphOut=args.graphOut, thresholdStep=args.steps)
    
