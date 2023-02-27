import pandas as pd

def separateOutliers(inData, out = None, inputType = "df", outliersOut = None, returnOutliers = False):
    
    # If input is a csv path read it into a df, if df copy, else report error
    if inputType == "csv":
        inDf = pd.read_csv(inData)
        print("Loaded from csv")
    elif inputType == "df":
        inDf = inData.copy()
        print("Using data frame")
    else:
        print("Enter a valid input type (df or csv). You entered")
        print(inputType)
        return
    
    # Make sure topic column has ints not strings
    inDf["Topic"] = inDf["Topic"].astype("int32")

    # Split df into outliers and nonoutliers
    nonOutliers = inDf[inDf["Topic"] != -1]
    outliers = inDf[inDf["Topic"] == -1]
    print("Outliers split... exporting")

    # Return according to specified params
    if out:
        nonOutliers.to_csv(out, index=False, encoding='utf-8-sig')
    if outliersOut:
        outliers.to_csv(outliersOut, index=False, encoding='utf-8-sig')
    
    if returnOutliers:
        return nonOutliers, outliers
    else:
        return nonOutliers



    
        
