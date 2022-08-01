from encoders.base import Encoder
from sentence_transformers import SentenceTransformer


class BERT(Encoder):
    def __init__(self, model_path: str = "bert-base-nli-mean-tokens"):
        self.model = SentenceTransformer(model_path)

    # @property
    def encode(self, sentence):
        embeddings = self.model.encode([sentence])
        return embeddings[0]

    def encode_array(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings