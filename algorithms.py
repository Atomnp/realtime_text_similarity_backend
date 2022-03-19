from importlib.resources import path
from encoders.arora import Arora
from encoders.use import UniversalEncoder
from encoders.bert import BERT
from encoders.word2vec import word2vec
from enum import Enum
from search_index import AnnoyIndex


class Algorithm(Enum):
    WORD_2_VEC = 1
    USE = 2
    ARORA = 3
    BERT = 4


class Runtime:
    def __init__(self):
        self.current_algo = Algorithm.USE
        self.encoder = UniversalEncoder(model_path="./USE")
        #  we need to know the dimension of vectors embedding to load existing index
        annoy_index = AnnoyIndex(dimension=512)
        annoy_index.load("./indices/sent_enc_index.ann")
        self.index = annoy_index
        self.encoder_type = Algorithm.USE

    def get_similar(self, question):
        emb = self.encoder.encode(question)
        return self.index.query_cosine(emb)

    # generates new index from uploaded data
    def change_index(self, filename):
        questions = []
        with open(filename, "r") as fp:
            questions = [
                x.strip().lower().split("?,") for x in fp.readlines() if x != "\n"
            ]
        questions_string = [question[0] for question in questions]
        embeddings = self.encoder.encode_array(questions_string)
        annoy_index = AnnoyIndex(dimension=len(embeddings[0]))
        annoy_index.build(embeddings, questions_string)
        annoy_index.save("./indices/" + filename + str(self.current_algo) + ".ann")
        self.index = annoy_index

    def switch_algo(self, algo: Algorithm, filename="default"):
        self.current_algo = algo
        if self.encoder_type == algo:
            return
        elif algo == Algorithm.USE:
            self.encoder = UniversalEncoder(model_path="./USE")
            #  we need to know the dimension of vectors embedding to load existing index
            annoy_index = AnnoyIndex(dimension=512)
            annoy_index.load("./indices/sent_enc_index.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.USE
        elif algo == Algorithm.WORD_2_VEC:
            self.encoder = word2vec()
            annoy_index = AnnoyIndex(dimension=100)
            annoy_index.load("./indices/word2vec.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.WORD_2_VEC
        elif algo == Algorithm.ARORA:
            self.encoder = Arora()
            annoy_index = AnnoyIndex(dimension=100)
            annoy_index.load("./indices/arora.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.ARORA
        elif algo == Algorithm.BERT:
            self.encoder = BERT()
            annoy_index = AnnoyIndex(dimension=768)
            annoy_index.load("./indices/bert.ann")
            self.index = annoy_index
            self.encoder_type = Algorithm.BERT
