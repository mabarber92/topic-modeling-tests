import pandas as pd
import sys

def mergeTopicsCsv(primaryCsv, secondaryCsv, csvOut, idCols, sentenceCol="phrase"):
    print(idCols)
    # Depending on whether list is given create the list that will be used to select columns for the merge 
    if not type(idCols) == list:
        selectCols = ["Topic", idCols]
        idCols = [idCols]
        if sentenceCol:
            selectCols.append(sentenceCol)
    else:
        selectCols = ["Topic"] + idCols
        if sentenceCol:
            selectCols.append(sentenceCol)
    
    # Load in data - only select columns needed from the secondary Df
    primaryDf = pd.read_csv(primaryCsv)
    secondaryDf = pd.read_csv(secondaryCsv)[selectCols]
    if sentenceCol:
        selectCols.remove(sentenceCol)
    
    # If sentence column is not None it will be used to help build the unique identifier for the merge
    if sentenceCol:
        primaryDf["idCol"] = primaryDf[sentenceCol].str[:2]
        secondaryDf["idCol"] = secondaryDf[sentenceCol].str[:2]
    else:
        primaryDf["idCol"] = ""
        secondaryDf["idCol"] = ""

    # Create the unique identifier
    for col in idCols:
        primaryDf["idCol"] = primaryDf["idCol"] + "." + primaryDf[col].astype(str)
        secondaryDf["idCol"] = secondaryDf["idCol"] + "." + secondaryDf[col].astype(str)
    print(primaryDf["idCol"])
    print(secondaryDf["idCol"])

    secondaryDf = secondaryDf.drop(columns=idCols + [sentenceCol])
    # Merge on the unique id - ensuring we get an old and a new topic id
    primaryDf = primaryDf.merge(secondaryDf, left_on='idCol', right_on = "idCol", how='left', suffixes=("", "Run1"))
    
    print(primaryDf.head().transpose())
    # Clean up
    primaryDf = primaryDf.drop(columns=["idCol"])
    
    # Export
    primaryDf.to_csv(csvOut, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    print("Enter number of columns to merge into Id")
    columnNo = input()
    inputCols = []
    for i in range(0, int(columnNo)):
        print("Enter the {} column".format(str(i)))
        column = input()
        inputCols.append(column)
    print(inputCols)
    mergeTopicsCsv(sys.argv[1], sys.argv[2], sys.argv[3], inputCols)

