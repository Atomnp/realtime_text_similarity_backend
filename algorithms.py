from importlib.resources import path
from encoders.arora import Arora
from encoders.use import UniversalEncoder
from encoders.bert import BERT
from enum import Enum
from search_index import AnnoyIndex


class Algorithm(Enum):
    WORD_2_VEC = 1
    USE = 2
    ARORA = 3
    BERT = 4


class Runtime:
    def __init__(self):
        self.encoder = UniversalEncoder(model_path="./USE")
        #  we need to know the dimension of vectors embedding to load existing index
        annoy_index = AnnoyIndex(dimension=512)
        annoy_index.load("./indices/sent_enc_index.ann")
        self.index = annoy_index
        self.encoder_type = Algorithm.USE

    def get_similar(self, question):
        emb = self.encoder.encode(question)
        return self.index.query(emb)

    def switch_algo(self, algo: Algorithm):
        if self.encoder_type == algo:
            return
        if algo == Algorithm.WORD_2_VEC:
            # self.encoder = word2vecencoder
            # self.encoder
            pass
        elif algo == Algorithm.ARORA:
            self.encoder = Arora()
            annoy_index = AnnoyIndex(dimension=100)
            annoy_index.load("./indices/weighted_annoy_index.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.ARORA
        elif algo == Algorithm.BERT:
            self.encoder = BERT()
            annoy_index = AnnoyIndex(dimension=768)
            annoy_index.load("./indices/bert.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.BERT
