# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:21:32 2022

@author: mathe
"""

import re
import pandas as pd
from math import ceil
from openiti.helper.funcs import text_cleaner
import os

def changePars(maxCh, seqCap, minSplit):
    adaptiveSplit = False
    print("\n---------\n")
    print("Number of splits too small for the sequence cap. Minimum split number: ")
    minSplitCap = maxCh/seqCap
    print(minSplitCap)
    print("Minimum sequence cap:")
    seqCapCap = maxCh/minSplit
    print(seqCapCap)
    print("\n---------\n")
    print("Choose to do one or all of the following: \n 1. Change split number \n 2. Change sequence cap \n 3. Adaptive split (for each milestone use the minimum number of splits possible for the sequence cap).")
    print("Choose new split number: (minimum number of splits for current cap: " + str(minSplitCap) + ")")
    minSplit = int(input())
    seqCapCap = maxCh/minSplit
    print("Choose a new sequence cap: (minimum sequence cap for your chosen split number: " + str(seqCapCap) + ")")
    seqCap = int(input())
    print("Would you prefer to use an adaptive split for this sequence cap? (0 for no and 1 for yes)")
    choice = input()
    if choice == "1":
        adaptiveSplit = True
    else:
        adaptiveSplit = False
    
    return seqCap, minSplit, adaptiveSplit

def InputFromText(textPath, outPath = None, seqCap = 512, minSplit = 2, adaptiveSplit = False, uriCol = False, shingle=False):
    """Take Arabic source text split it on milestones and then split based
    on the specified number of splits. seqCap specifies the maximum length 
    in chars of the sequence. The default value is the maximum sequence length
    accepted for a Bert embedding. It will first check if the text can be
    split into the number of splits without hitting the sequence cap. If it hits
    the cap then it will suggest an alternative splitting count and ask 
    for confirmation. Adaptive split instead creates the optimum split for
    each milestone, ensuring the sequent cap is not exceeded"""
    
    if type(textPath) == list:
        if len(textPath) > 1:
            print("Multiple texts detected - adding URI column to output")
            uriCol = True
    else:
        textPath = [textPath]
    
    msDictList = []
    
    for path in textPath:
        
        with open(path, encoding ='utf-8-sig') as f:
            text = f.read()
            f.close()
            
        text = re.split("#META#Header#End#", text)[-1]
        
        # Create a df containing milestones and corresponding text
        msSplits = re.split(r"ms(\d+)", text)
        totalSplits = len(msSplits)
        
        if uriCol:
            uri = path.split("\\")[-1]
            for idx, split in enumerate(msSplits):
                if not re.match(r"\d", split) and idx+1 != totalSplits:
                    msDictList.append({"ms": msSplits[idx+1], "text": split, "chLength": len(split), "uri": uri})
        else:
            for idx, split in enumerate(msSplits):
                if not re.match(r"\d", split) and idx+1 != totalSplits:
                    msDictList.append({"ms": msSplits[idx+1], "text": split, "chLength": len(split)})
    
    msDf = pd.DataFrame(msDictList)
    
    # Check that the supplied parameters will work for the text
    maxCh = msDf["chLength"].max()
    while maxCh/minSplit > seqCap and not adaptiveSplit:
        seqCap, minSplit, adaptiveSplit = changePars(maxCh, seqCap, minSplit)
    
    print(seqCap, minSplit, adaptiveSplit)
    
    # Clean up df
    del msDf
        
    
    # Loop through ms dict to create relevant splits
    outDictList = []
    for ms in msDictList:
        
        
        if adaptiveSplit:
            splitCount = ceil(ms["chLength"]/seqCap)
            msText = text_cleaner(ms["text"])
            tokens = msText.split()
            
            tokCount = len(tokens)
            
            tokWidth = ceil(tokCount/splitCount)
            iterCap = splitCount
        else:
            tokens = ms["text"].split()
            
            tokCount = len(tokens)
            
            tokWidth = ceil(tokCount/minSplit)
            iterCap = minSplit
        pos = 0
        splitid = 0
        remainingToks = tokCount
        finalIt = False
        
        while remainingToks > 0:            
            
            
            lastTok = pos+tokWidth
            split = " ".join(tokens[pos:lastTok])
            
            while len(split) > seqCap:
                lastTok = lastTok - 1
                split = " ".join(tokens[pos:lastTok])
            if len(split) != 0 and not re.match(r"\s+", split):
                if uriCol:
                    outDictList.append({"ms": ms["ms"], "split": splitid, "text": split, "uri": ms["uri"]})
                else:
                    outDictList.append({"ms": ms["ms"], "split": splitid, "text": split})
            if shingle:
                pos = lastTok - ceil(tokWidth/2)
            else:
                pos = lastTok
            remainingToks = tokCount - pos
            splitid = splitid + 1
            
            if remainingToks < tokWidth:
                
                split = " ".join(tokens[-remainingToks:])                
                if len(split) > seqCap:
                    
                    split1 = " ".join(tokens[-remainingToks:-ceil(remainingToks/2)])
                    
                        
                    split2 = " ".join(tokens[-ceil(remainingToks/2)])
                   
                        
                    if len(split1) != 0 and not re.match(r"\s+", split1):
                        if uriCol:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split1, "uri": ms["uri"]})
                        else:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split1})
                    if len(split2) != 0 and not re.match(r"\s+", split2):
                        splitid = splitid + 1
                        if uriCol:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split1, "uri": ms["uri"]})
                        else:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split2})
                    if len(split1) > seqCap or len(split2) > seqCap:
                        print("WARNING: a sequence exceeds your specified cap")
                    remainingToks = 0
                else:
                    if len(split) != 0 and not re.match(r"\s+", split):
                        if uriCol:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split, "uri": ms["uri"]})
                        else:
                            outDictList.append({"ms": ms["ms"], "split": splitid, "text": split})
                    remainingToks = 0
           
                
            
        # else:
            
        #     tokens = ms["text"].split()
            
        #     tokCount = len(tokens)
            
        #     tokWidth = ceil(tokCount/minSplit)
        #     pos = 0
        #     for i in range(0, minSplit):
        #         remainingToks = tokCount - pos
        #         if remainingToks < tokWidth:
        #             split = " ".join(tokens[-remainingToks:])
        #             if len(split) > seqCap:
                        
        #                 split1 = " ".join(tokens[-remainingToks:-ceil(remainingToks/2)])
        #                 split2 = " ".join(tokens[-ceil(remainingToks/2)])
        #                 if uriCol:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split1, "uri": ms["uri"]})
        #                 else:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split1})
        #                 if uriCol:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split2, "uri": ms["uri"]})
        #                 else:
        #                     outDictList.append({"ms": ms["ms"], "split": i+1, "text": split2})
        #                 if len(split1) > seqCap or len(split2) > seqCap:
        #                     print("WARNING: a sequence exceeds your specified cap")
        #             else:
        #                 if uriCol:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split, "uri": ms["uri"]})
        #                 else:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split})
                
        #         else:
        #             lastTok = pos+tokWidth
        #             split = " ".join(tokens[pos:lastTok])
        #             while len(split) > seqCap:
        #                 lastTok = lastTok - 1
        #                 split = " ".join(tokens[pos:lastTok])
        #             if uriCol:
        #                     outDictList.append({"ms": ms["ms"], "split": i, "text": split, "uri": ms["uri"]})
        #             else:
        #                 outDictList.append({"ms": ms["ms"], "split": i, "text": split})
        #             pos = lastTok
    
    outDf = pd.DataFrame(outDictList)
    if outPath is not None:
        outDf.to_csv(outPath, index=False, encoding='utf-8-sig')
    return outDf                
        
   
path = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/corpus/0845Maqrizi.Mawaciz.MAB02082022-ara1.completed"
outPath = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/BERTopic/MaqriziCorpus-ShinglingTests/256-adaptiveSplit.csv"

inDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/corpus/MaqriziCorpus"
path = []
for root, dirs, files in os.walk(inDir, topdown=False):
    for name in files:
        path.append(os.path.join(root, name))

print(path)

out = InputFromText(path, outPath, seqCap=256, adaptiveSplit=True, shingle=True)
    
                               