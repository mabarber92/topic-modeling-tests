from InputFromText import InputFromText
from BERTopic_model import csvToBERTopic
import pandas as pd
import os
from tqdm import tqdm
from multiprocess import Pool



def createSamplefromOpenITI(metadata_path, corpus_base_path, dateRange = [1,1000], dateSlice = 100, samplingStrategy = [{"column": "tok_length", "samples": [{"bounds" : [0, 20000], "count": 4, "combine": True}, {"bounds" : [100000, 150000], "count": 1, "combine": False}, {"bounds" : [1000000, 4000000], "count": 1, "combine": False}]}]):
    """This function creates a sample of text addresses using a random sampling strategy but based on a set of parameters. For example, to take a random set of samples that belong to a set range of lengths sampling by each century"""
    # Load in the metadata, apply base level filtering, create the corpus base path column and strip out unneeded columns
    metadata = pd.read_csv(metadata_path, sep="\t")
    metadata = metadata[metadata["status"] == "pri"]
    metadata = metadata[metadata["date"].between(dateRange[0], dateRange[1])]
    metadata["rel_path"] = corpus_base_path + metadata["local_path"].str.split("/master/|\.\./", expand = True, regex=True)[1]
    
    print("Metadata loaded")

    # Initial list to populate
    samplePaths = []

    # Loop through specified date range to specified endpoint
    for d in tqdm(range(dateRange[0], dateRange[1], dateSlice)):
        # Fetch the date slice
        
        dfDateRanged = metadata[metadata["date"].between(d, d+dateSlice)]
        
        # Slice based on sampling stategies
        for strategy in samplingStrategy:
            for sample in strategy["samples"]:
                dfExtract = dfDateRanged[dfDateRanged[strategy["column"]].between(sample["bounds"][0], sample["bounds"][1])]
                
                # Fetch a random sample of rows
                if len(dfExtract) > 0:
                    sampleList = dfExtract.sample(n=sample["count"])["rel_path"].tolist()
                if len(sampleList) > 0:
                    if sample["combine"] == True:
                        samplePaths.append(sampleList)
                    else:
                        samplePaths.extend(sampleList)
    
    return samplePaths

def topicModelSamples(samplesList, seqLens = [256, 512, 1024] outlierThresholds = (0, 10), outlierTypes=['probabilities', 'c-tf-idf'] n_neighbours=[5,15,20,25,30]):

    # Set up pooling to initiate work
    p = Pool(4)

    # Initiate a embedding model for all work

    # Loop through seqLens
        # Loop through shingling and not
        # Create embedding for that set up splits
        #For each seqLens try each outlier type

            # Loop through each threshold

                # Loop through each n_neighbours

                    # Loop through texts and produce output

corpusbasePath = "D:/OpenITI Corpus/corpus_2022_1_6/"
metaPath = "D:/Corpus Stats/2022/OpenITI_metadata_2022-1-6.csv"

samples = createSamplefromOpenITI(metaPath, corpusbasePath)
print(samples)


