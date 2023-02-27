import pandas as pd
import re
from tqdm import tqdm

dfPath = "../../data/inputs/searchResults/largerWindow/results.csv"
outPath = "../../data/inputs/searchResults/largerWindow/DuplicatedResults.csv"
df = pd.read_csv(dfPath)
dictList = df.sort_values(by=["uri", "ms"]).to_dict("records")

simRows = []

for idx, item in enumerate(tqdm(dictList)):
    if len(re.findall("يوسف", item["phrase"])) > 1:
        simRows.extend(dictList[idx:idx+1])

outDf = pd.DataFrame(simRows)
outDf.to_csv(outPath, index=False, encoding='utf-8-sig')

