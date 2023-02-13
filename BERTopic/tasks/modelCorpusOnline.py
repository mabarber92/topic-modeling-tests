

from sentence_transformers import SentenceTransformer, models
from torch import nn
import os
from mainFuncs.BERTopic_model import csvToBERTopic
from mainFuncs.createModelOnline import createModelOnline
from torch import cuda


def modelCorpus(inDir, outDir, inputType="csvs", seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, modelSummary = None):
    
    # Intitate a model to loop around and update
    print("creating embedding model")
    model_name = embeddingModel
    max_seq_length= seqLength
    word_embedding_model = models.Transformer(model_name, max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                               out_features=seqLength, 
                               activation_function=nn.Tanh())
    transformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    device = "cuda:0" if cuda.is_available() else "cpu"
    print(device)
    transformerModel = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
    print("model loaded")
    
    # creatModelOnline will partial_fit() the model by iterating through the corpus and will create the csvs to provide outputs
    topicModel = createModelOnline(inDir, outDir, inputType=inputType, seqLength=seqLength, transformerModel=transformerModel, shingle=shingle)
    
    # Loop through the inDir updating the model with each text and transform
    for root, dirs, files in os.walk(outDir, topdown=False):
        for idx, name in enumerate(files):
            path = os.path.join(root, name)
            print(path)            
            csvToBERTopic(path, path, transformer=transformerModel, existingModel=topicModel, seqLength = seqLength)
                
           
             
    
    # If modelSummary return a csv containing a summary of the model's topics
    if modelSummary:
        summaryDf = topicModel.get_topic_info()
        summaryDf.to_csv(modelSummary, index=False, encoding='utf-8-sig')
    
    return topicModel    