import pandas as pd
from bertopic import BERTopic

from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from sentence_transformers import SentenceTransformer, models
from torch import nn
form torch import cuda
import os
from InputFromText import InputFromText
from tqdm import tqdm
from river import cluster
from river import stream


class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self


def createModelOnline(dirIn, dirOut=None, transformerModel=None, topicModel=None, n_clusters=500, inputType="csvs", seqLength=512, embeddingModel = "aubmindlab/bert-base-arabertv02", shingle=False, sentenceField="text"):
    
    # Intitate an embedding model if one isn't supplied
    if not transformerModel:
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
    
    # Initiate a topic model that is suitable for partial fitting
    if not topicModel:    
        print("Creating the topic model for partial fit")
        umap_model = IncrementalPCA(n_components=5)
        # cluster_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        cluster_model = River(cluster.DBSTREAM())
        vectorizer_model = OnlineCountVectorizer(decay=.01)
        
        topicModel = BERTopic(umap_model=umap_model,                    
                           vectorizer_model=vectorizer_model, hdbscan_model=cluster_model, language="multilingual", verbose=True)
    
    print("commencing Partial fit")
    for root, dirs, files in os.walk(dirIn, topdown=False):
        for name in tqdm(files):
            path = os.path.join(root, name)
            
            # Import the sentences for embedding and modeling - if texts are supplied split them
            if inputType == "texts":
                if shingle:
                    csvName = name + "-" + str(seqLength) + "-shingled-Topics.csv"
                else:
                    csvName = name + "-" + str(seqLength) + "-Topics.csv"
                outPath = os.path.join(dirOut, csvName)
                df = InputFromText(path, outPath, seqCap = seqLength, adaptiveSplit=True, shingle=shingle)
            if inputType == "csvs":
                df = pd.read_csv(path, encoding='utf-8-sig')
            
            # Embed the sentences
            sentences = df[sentenceField].tolist()
            embeds = transformerModel.encode(sentences)
            
            # Partially fit the model using the sentences
            topicModel.partial_fit(df[sentenceField], embeds)
    
    print("Partially fitted Model to Corpus")
    return topicModel