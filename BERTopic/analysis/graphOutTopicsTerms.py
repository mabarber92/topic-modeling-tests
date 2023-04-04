import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

def graphOutTopicsTerms(outTermCsv, topicList, focusTerms, graphOut):
    if type(topicList) is not list:
        topicList = [topicList]
    if type(focusTerms) is not list:
        focusTerms = [focusTerms]

    # Load in data
    outTermDf = pd.read_csv(outTermCsv)

    # Filter to only chosen topics
    outTermDf = outTermDf[outTermDf["Topic"].isin(topicList)]

    # Loop through foucus Terms to create a df for graphing
    
    termCompDf = pd.DataFrame()
    
    concatCols = list(outTermDf.columns.values)
    removeCols = ["Topic", "Count", "Outlier_thres", "Total_terms", "Percent_of_count"] + focusTerms
    for col in removeCols:
        if col in concatCols:
            concatCols.remove(col)
        else:
            print("{} not found in input csv columns".format(col))

    print(concatCols)

    outTermDict = outTermDf.to_dict("records")
    dictForGraph = []

    for row in outTermDict:
        
        for term in focusTerms:
            new_row = {"Topic": row["Topic"], "Outlier_thres": row["Outlier_thres"], "Term": term, "Count": row[term]}
            dictForGraph.append(new_row)
        concatSum = 0
        for col in concatCols:
            concatSum = concatSum + row[col]
        new_row = {"Topic": row["Topic"], "Outlier_thres": row["Outlier_thres"], "Term": "Other terms", "Count": concatSum}
        dictForGraph.append(new_row)

    dfForGraph = pd.DataFrame(dictForGraph)
    print(dfForGraph)

    # Use the new df to set up a facetgrid and output

    g = sns.FacetGrid(data=dfForGraph, col="Topic", col_wrap = 5, sharey=False, sharex=True)
    g.map(sns.lineplot, "Outlier_thres", "Count", "Term")        
    g.add_legend()

    new_terms = []
    for ltr_term in dfForGraph["Term"].drop_duplicates().to_list():
        if ltr_term != "Other terms":
            new_label = get_display(arabic_reshaper.reshape(ltr_term))
            new_terms.append(new_label)
        else:
            new_terms.append(ltr_term)
        
    for t, l in zip(g._legend.texts, new_terms):
        t.set_text(l)

    g.set(xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    g.set_xticklabels(["0","", "0.2", "", "0.4","", "0.6","", "0.8","", "1"])

    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    figure = g.fig       
    figure.savefig(graphOut, dpi=300, bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    graphingPars = pd.read_csv(sys.argv[3])
    focusTerms = graphingPars["Terms"].dropna().to_list()
    topicList = graphingPars["Topics"].dropna().to_list()
    
    print(focusTerms)
    print(topicList)

    graphOutTopicsTerms(sys.argv[1], topicList, focusTerms, sys.argv[2])
