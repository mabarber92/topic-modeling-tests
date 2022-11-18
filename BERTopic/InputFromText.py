# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:21:32 2022

@author: mathe
"""

import re
import pandas as pd
from math import ceil

def changePars(maxCh, seqCap, minSplit):
    adaptiveSplit = False
    print("Number of splits too small for the sequence cap. Minimum split number: ")
    minSplitCap = maxCh/seqCap
    print(minSplitCap)
    print("Minimum sequence cap:")
    seqCapCap = maxCh/minSplit
    print(seqCapCap)
    print("\n---------\n")
    print("Choose to \n 1. Change split number \n 2. Change sequence cap \n 3. Adaptive split (for each milestone use the minimum number of splits possible for the sequence cap Enter 1, 2 or 3.")
    choice = input()
    print(choice)
    if choice == "1." or choice == "1":
        
        print("Choose new split number:")
        minSplit = int(input())
        while minSplit < minSplitCap:
            print("Number of splits too low. Choose a higher count")
            minSplit = int(input())
    if choice == "2." or choice == "2":
        print("Choose new sequence cap:")
        seqCap = int(input())
        while seqCap < seqCapCap:
            print("Length of sequence too low. Choose a higher count")
            seqCap = int(input())
    if choice == "3." or choice == "3":
        adaptiveSplit = True
    return seqCap, minSplit, adaptiveSplit

def InputFromText(textPath, outPath, seqCap = 512, minSplit = 2, adaptiveSplit = False):
    """Take Arabic source text split it on milestones and then split based
    on the specified number of splits. seqCap specifies the maximum length 
    in chars of the sequence. The default value is the maximum sequence length
    accepted for a Bert embedding. It will first check if the text can be
    split into the number of splits without hitting the sequence cap. If it hits
    the cap then it will suggest an alternative splitting count and ask 
    for confirmation. Adaptive split instead creates the optimum split for
    each milestone, ensuring the sequent cap is not exceeded"""
    
    with open(textPath, encoding ='utf-8-sig') as f:
        text = f.read()
        f.close()
        
    text = re.split("#META#Header#End#", text)[-1]
    
    # Create a df containing milestones and corresponding text
    msSplits = re.split(r"ms(\d+)", text)
    totalSplits = len(msSplits)
    msDictList = []
    
    for idx, split in enumerate(msSplits):
        if not re.match(r"\d", split) and idx+1 != totalSplits:
            msDictList.append({"ms": msSplits[idx+1], "text": split, "chLength": len(split)})
    
    msDf = pd.DataFrame(msDictList)
    
    # Check that the supplied parameters will work for the text
    maxCh = msDf["chLength"].max()
    if maxCh/minSplit > seqCap and not adaptiveSplit:
        seqCap, minSplit, adaptiveSplit = changePars(maxCh, seqCap, minSplit)
    
    print(seqCap, minSplit, adaptiveSplit)
    
    # Clean up df
    del msDf
    
    # Loop through ms dict to create relevant splits
    outDictList = []
    for ms in msDictList:
        
        if adaptiveSplit:
            splitCount = ceil(ms["chLength"]/seqCap)
            tokens = ms["text"].split()
            
            tokCount = len(tokens)
            
            tokWidth = ceil(tokCount/splitCount)
            pos = 0
            for i in range(0, splitCount):
                remainingToks = tokCount - pos
                if remainingToks < tokWidth:
                    split = " ".join(tokens[-remainingToks:])
                    if len(split) > seqCap:
                        
                        split1 = " ".join(tokens[-remainingToks:-ceil(remainingToks/2)])
                        split2 = " ".join(tokens[-ceil(remainingToks/2)])
                        outDictList.append({"ms": ms["ms"], "split": i, "text": split1})
                        outDictList.append({"ms": ms["ms"], "split": i+1, "text": split2})
                    else:
                        outDictList.append({"ms": ms["ms"], "split": i, "text": split})
                
                else:
                    lastTok = pos+tokWidth
                    split = " ".join(tokens[pos:lastTok])
                    while len(split) > seqCap:
                        lastTok = lastTok - 1
                        split = " ".join(tokens[pos:lastTok])
                    outDictList.append({"ms": ms["ms"], "split": i, "text": split})
                    pos = lastTok
    
    outDf = pd.DataFrame(outDictList)
    return outDf                
        
    
path = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/corpus/0310Tabari.Tarikh.Shamela0009783BK1-ara1.mARkdown"
out = InputFromText(path, "")
    
                               