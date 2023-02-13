from mainFuncs.InputFromText import InputFromText
from mainFuncs.BERTopic_model import csvToBERTopic
from sentence_transformers import SentenceTransformer, models
from torch import nn
import torch
from multiprocess import Pool
import os
import pandas as pd
from tqdm import tqdm
import time

# def check_cuda(model):
# 	if torch.cuda.is_available():
# 		cuda_device = "cuda:0"
# 		model = model.to(cuda_device)
# 	else:
#         cuda_device = "cpu"

# 	return model, cuda_device

def createInputAndModel(inPath, outPath, summaryPath, seqLength, returnSummary, transformerModel, existingCsv = True):
    print(inPath)
    if not existingCsv:
        df = InputFromText(inPath, seqCap = seqLength, adaptiveSplit=True, shingle=True)
    else:
        df = pd.read_csv(inPath)
    if len(df) < 15:
        print("Text too small to topic model (less than 15 rows) - consider grouping texts")
    else:
        topicSummary = csvToBERTopic(df, csvOut=outPath, inputType="df",  seqLength = seqLength, embeddingModel = "aubmindlab/bert-base-arabertv02", transformer = transformerModel, returnSummary=returnSummary)
        topicSummary.to_csv(summaryPath, index=False, encoding='utf-8-sig')           

def modelEachTextInCorpus(outDir, metadata_path, corpus_base_path=None, inCsvDir = None, end_date = 1000, 
                       seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, modelSummary = 10, overwrite=False, exceptionLength=1000000):
    
    # Set up directories for output
    topicsOut = os.path.join(outDir, "topicsByText")
    if not os.path.exists(topicsOut):
        os.mkdir(topicsOut)
    summariesOut = os.path.join(outDir, "summaries")
    if not os.path.exists(summariesOut):
        os.mkdir(summariesOut)
    mainSummary = os.path.join(outDir, "fullSummary.csv")
    
    # Load in metadata
    metadata = pd.read_csv(metadata_path, sep = "\t")
    
    # Create list to store locations to return to without multiprocessing
    
    returnTo = []
    returnToCsv = []

    
    
    # Initiate embedding model
    print("creating embedding model")
    model_name = embeddingModel
    max_seq_length= seqLength
    word_embedding_model = models.Transformer(model_name, max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                               out_features=seqLength, 
                               activation_function=nn.Tanh())
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    transformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
    # transformerModel, cuda = check_cuda(transformerModel)
    print("model loaded")


    # If there are no in csvs already we need to fetch the texts   
    if not inCsvDir:
        inCsvDir = outDir

        # Create a relative path for the texts using specified corpus base path
        metadata["rel_path"] = corpus_base_path + metadata["local_path"].str.split("/master/|\.\./", expand = True, regex=True)[1]
        
        # Create exceptions list for files that are too large to be multiprocessed
        exceptions = metadata[metadata["tok_length"] > exceptionLength]["rel_path"].to_list()   
        
        # Fetch all of the texts and filter        
        metadata = metadata[metadata["status"] == "pri"]
        metadata = metadata[metadata["date"] <= end_date]
        
        location_list = metadata["rel_path"].to_list()   

        # Set up parralel processing
        p = Pool(processes=4, maxtasksperchild=1000)

        # Loop through text paths create topic and summary csvs      
        for location in tqdm(location_list):
            if os.path.exists(location):
                if location in exceptions:
                    print("Adding " + location + " to return to w/o multiprocessing" )
                    returnTo.append(location)
                else:
                    uri = ".".join(location.split("/")[-1].split("-")[0:-1])
                    
                    outPath = topicsOut + "/" + uri + ".csv"
                    # Skipping files that already exist in outlocation
                    if os.path.isfile(outPath) and not overwrite:
                        print("Already exists... skipping...")
                        continue
                    summaryPath = summariesOut + "/" + uri + ".csv"                
                    p.apply_async(createInputAndModel, args=(location, outPath, summaryPath, seqLength, modelSummary, transformerModel, False))
                          
            else:
                print(location + " - path not found")
        
        print("Csv files created")
    else:
        print("Using existing list of csvs")
        # Locate exceptions
        exceptions = metadata[metadata["tok_length"] > exceptionLength]["version_uri"].str.split("-", expand=True)[0].to_list()
        
        # Set up parralel processing
        p = Pool(4)

        # Loop through text paths create topic csvs and concatenate a summary csv
        for root, dirs, files in os.walk(inCsvDir, topdown=False):
            for name in tqdm(files):
                location = os.path.join(root, name)
                if name in exceptions:
                    print("Adding " + name + " to return to w/o multiprocessing" )
                    returnToCsv.append(location)
                else:
                    outPath = topicsOut + "/" + name + ".csv"
                    # Skipping files that already exist in out location if not overwrite
                    if os.path.isfile(outPath) and not overwrite:
                        print("Already exists... skipping...")
                        continue
                    summaryPath = summariesOut + "/" + name + ".csv"  
                    p.apply_async(createInputAndModel, args=(location, outPath, summaryPath, seqLength, modelSummary, transformerModel, True))

    # End parrallel processing            
    p.close()
    p.join()
    
    # Go through return-to lists
    print("Processing texts above specified maximum length")
    for location in returnTo:
        name = ".".join(location.split("/")[-1].split("-")[0:-1])
        outPath = topicsOut + "/" + name + ".csv"
        # Skipping files that already exist in out location if not overwrite
        if os.path.isfile(outPath) and not overwrite:
            print("Already exists... skipping...")
            continue
        summaryPath = summariesOut + "/" + name + ".csv"
        createInputAndModel(location, outPath, summaryPath, seqLength, modelSummary, transformerModel, False)
    
    for location in returnToCsv:
        name = ".".join(location.split("/")[-1].split("-")[0:-1])
        outPath = topicsOut + "/" + name + ".csv"
        # Skipping files that already exist in out location if not overwrite
        if os.path.isfile(outPath) and not overwrite:
            print("Already exists... skipping...")
            continue
        summaryPath = summariesOut + "/" + name + ".csv"
        createInputAndModel(location, outPath, summaryPath, seqLength, modelSummary, transformerModel, True)

    # Concatenate all those in summary
    print("Concatenating the topic summary csv...")
    summaryDf = pd.DataFrame()
    for root, dirs, files in os.walk(summariesOut, topdown=False):
        for name in tqdm(files):
            csvPath = os.path.join(root, name)
            df = pd.read_csv(csvPath)
            df["uri"] = name.split(".cs")[0]
            summaryDf = pd.concat([summaryDf, df])
    
    summaryDf.to_csv(mainSummary, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    startTime = time.time()
    corpus_base_path = "D:/OpenITI Corpus/corpus_2022_1_6/"
    metadata_path = "D:/Corpus Stats/2022/OpenITI_metadata_2022-1-6.csv"
    outDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus256/"
    modelEachTextInCorpus(outDir, metadata_path, corpus_base_path,  end_date=1000, seqLength=256)
    duration = time.time() - startTime
    print("Time to run: " + str(duration))