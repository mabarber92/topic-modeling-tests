from sentence_transformers import SentenceTransformer, models
from torch import nn
from torch import cuda

def initialiseEmbedModel(model_name, seqLength=512):
    print("loading model...")       
    word_embedding_model = models.Transformer(model_name, seqLength)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True)
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                            out_features=seqLength, 
                            activation_function=nn.Tanh())
    # It seems we may not need the device set up
    device = "cuda:0" if cuda.is_available() else "cpu"
    print(device)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device = device)
    print("model loaded")
    return model