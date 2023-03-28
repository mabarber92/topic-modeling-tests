import re
import sys
import pandas as pd
from tqdm import tqdm

def regexCheckTops(dataIn, regexList, outAnalysisCsv = None, csvIn = True):
    if csvIn:
        topDf = pd.read_csv(dataIn)
    else:
        topDf = dataIn.copy()
    
    topicList = topDf[["Topic", "Count"]].drop_duplicates().values.tolist()
    
    if type(regexList) is not list:
        regexList = [regexList]
    
    outDictList = []
    for topic in tqdm(topicList):
        filteredDf = topDf[topDf["Topic"] == int(topic[0])]
        dictOut = {"Topic": topic[0], "Count": topic[1]}
        for regex in regexList:
            count = len(filteredDf[filteredDf["phrase"].str.contains(regex)])
            dictOut[regex] = count
        outDictList.append(dictOut)

    outDf = pd.DataFrame(outDictList)
    if outAnalysisCsv:
        outDf.to_csv(outAnalysisCsv, index=False, encoding = 'utf-8-sig')
    
    return outDf

if __name__ == "__main__":
    print("Starting..")
    print(str(sys.argv))
    regexCheckTops(sys.argv[1], ["ابو يوسف", "حجاج ا?بن يوسف"], sys.argv[2])