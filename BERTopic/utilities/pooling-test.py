from multiprocess import Pool
import pandas as pd
import os

def writeCsv(csvIn, csvOut):
    print(csvIn)
    print(csvOut)
    df = pd.read_csv(csvIn)
    df["New-col"] = "New data"
    df.to_csv(csvOut)

def runInParrallel():
    csvLoc = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus/"
    csvOutDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/poolingTest/"
    p = Pool()
    for root, dirs, files in os.walk(csvLoc, topdown=False):
        for name in files:
            csvIn = os.path.join(root, name)        
            csvOut = os.path.join(csvOutDir, name)
            p.apply_async(writeCsv, args = (csvIn, csvOut))
    p.close()
    p.join()

if __name__ == '__main__':
    runInParrallel()


