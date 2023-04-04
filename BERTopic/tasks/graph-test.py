import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from tqdm import tqdm

resultsDf = pd.read_csv("C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/outputs/searchModelling/outlierComp/changed-topics.csv")
compTermsDf = pd.read_csv("C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/outputs/searchModelling/compterms.csv")
returnDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/data/outputs/searchModelling/outlierComp/"
compTerms = compTermsDf["Term"].to_list()

sns.set(rc={'figure.figsize':(11.7,8.27)})

# Add an output directory for the graphs
graphDir = os.path.join(returnDir, "graphs/")
if not os.path.exists(graphDir):
    os.mkdir(graphDir)

# Loop through the term groups, produce a graph for each term group
for termGroup in compTermsDf["Group"].to_list():
    graphPath = os.path.join(graphDir, str(termGroup))
    if not os.path.exists(graphPath):
        os.mkdir(graphPath)
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

        # for outThres in termFilteredTopicsDf["Outlier_thres"].drop_duplicates().to_list():
            
        #     thresFiltered = termFilteredTopicsDf[termFilteredTopicsDf["Outlier_thres"] == outThres]
            
        #     summaryRow = thresFiltered.iloc[[0]]
        #     summaryRow["Term"] = term
        #     summaryRow["Topic"] = "Total Topics"
        #     summaryRow["_count"] = len(thresFiltered["Topic"].drop_duplicates())
        #     summaryRow["Type"] = "Topic Count"
            
        #     termCompDf = pd.concat([termCompDf, summaryRow])


    
    
    

    
    
    # 
    #     termFiltered["Term"] = "No Terms Found"
        
    #     termFiltered["_count"] = termFiltered["Total_terms"]
    #     termFiltered["Type"] = "Term Count"
    #     termCompDf = pd.concat([termCompDf, termFiltered])

    #     termFiltered["_count"] = termFiltered["Count"]
    #     termFiltered["Type"] = "Topic Length"
    #     termCompDf = pd.concat([termCompDf, termFiltered])
    topTopic = termCompDf["Topic"].max()

    for i in tqdm(range(0, topTopic, 25)):
        subGraphPath = graphPath + "/-" + str(i+25) + "-top-term-comp.png"
        topicList = list(range(i, i+25))
        print(topicList)
        termComp = termCompDf[termCompDf["Topic"].isin(topicList)]
        termComp["Topic"] = termComp["Topic"].astype('category')
        print(termComp["Topic"].drop_duplicates().to_list())    

        no_terms = len(compTerms)
        if no_terms > 5:
            col_wrap = 5
        else:
            col_wrap = no_terms
        g = sns.FacetGrid(data=termComp, row="Topic", col="Type", sharey=False, sharex=True)
        g.map(sns.lineplot, "Outlier_thres", "_count", "Term")        
        g.add_legend()

        new_terms = []
        for ltr_term in termComp["Term"].drop_duplicates().to_list():
            new_label = get_display(arabic_reshaper.reshape(ltr_term))
            new_terms.append(new_label)
        
        for t, l in zip(g._legend.texts, new_terms):
            t.set_text(l)


        print(g)

        for (row_val, col_val), ax in g.axes_dict.items():
            
            
            ax.set_title(col_val)
            ax.set_ylabel(row_val, rotation='vertical', horizontalalignment ='right')
            # if col_val == "Term Count":
            #     new_label = get_display(arabic_reshaper.reshape(row_val))
            #     ax.set_ylabel(new_label, rotation='horizontal', horizontalalignment ='right')
            # else:
            #     ax.set_ylabel("", rotation='horizontal', horizontalalignment ='right')


        figure = g.fig       
        figure.savefig(subGraphPath, dpi=300, bbox_inches='tight')
        plt.clf()

        # Remove rows that contain Topic of "Total Topics" - as messy for these graphs
        termComp["Topic"] = termComp["Topic"].astype('str') 
        termComp = termComp[termComp["Topic"] != "Total Topics"]
        termComp["Topic"] = termComp["Topic"].astype('category') 

        sns.set(rc={'figure.figsize':(11.7,8.27)})
            
        g = sns.lineplot(data=termComp, x="Outlier_thres", y="Total_terms", hue="Topic")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        figure = g.get_figure()
        figure.savefig(graphPath + "-absoluteterms.png", dpi=300, bbox_inches='tight')
        plt.clf()

        g = sns.lineplot(data=termComp, x="Outlier_thres", y="Percent_of_count", hue="Topic")
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        figure = g.get_figure()
        figure.savefig(graphPath + "-perc.png", dpi=300, bbox_inches='tight')
        plt.clf()


# Create a topic count graph for topics that have no terms identified

    

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