from encoders import *
from enum import Enum
from search_index import AnnoyIndex


class Algorithm(Enum):
    WORD_2_VEC = 1
    USE = 2
    ARORA = 3
    BERT = 4


class Runtime:
    def __init__(self):
        self.encoder = UniversalEncoder()
        annoy_index = AnnoyIndex()
        self.index = annoy_index.load("GIVE_ME_INDEX_PATH")
        self.encoder_type = Algorithm.USE

    def get_similar(self, question):
        emb = self.encoder.encode(question)
        return self.index.query(emb[0])

    def switch_algo(self, algo: Algorithm):
        if self.encoder_type != algo:
            return
        if algo == Algorithm.WORD_2_VEC:
            # self.encoder=
            pass
