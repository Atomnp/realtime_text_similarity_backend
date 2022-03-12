from gensim.models import Word2Vec
from encoders.base import Encoder
import numpy as np
from scipy.sparse import coo_matrix
from nltk.tokenize import word_tokenize
import nltk
import pickle

nltk.download("punkt")
nltk.download("stopwords")


class word2vec(Encoder):
    def __init__(self):
        self.w2v_model = Word2Vec.load("./indices/word2vec.model")
        self.vect = pickle.load(open("./indices/tf_vectorizer.pickle", "rb"))
        self.fv = self.vect.get_feature_names()

    def encode(self, sentence: str, embedding_size=100):
        to_transform = word_tokenize(sentence)
        matrix = self.vect.transform([to_transform])
        cx = coo_matrix(matrix)

        sorted_by_tfidf = sorted(
            [(self.fv[j], v) for i, j, v in zip(cx.row, cx.col, cx.data)],
            key=lambda x: x[1],
            reverse=True,
        )
        filtered = list(filter(lambda x: x[0] in self.w2v_model.wv, sorted_by_tfidf))

        arrlist = np.array(list(map(lambda x: self.w2v_model.wv[x[0]], filtered[:5])))

        sentence_embedding = np.mean(arrlist, axis=0)

        if type(sentence_embedding) == np.ndarray:
            return sentence_embedding
        else:
            return np.random.randn(embedding_size)

    def encode_array(self, sentences: str, embedding_size=100):
        embeddings = []
        for sentence in sentences:
            to_transform = word_tokenize(sentence)
            matrix = self.vect.transform([to_transform])
            cx = coo_matrix(matrix)

            sorted_by_tfidf = sorted(
                [(self.fv[j], v) for i, j, v in zip(cx.row, cx.col, cx.data)],
                key=lambda x: x[1],
                reverse=True,
            )
            filtered = list(
                filter(lambda x: x[0] in self.w2v_model.wv, sorted_by_tfidf)
            )
            if filtered == []:
                embeddings.append(np.random.randn(embedding_size))
                break
            arrlist = np.array(
                list(map(lambda x: self.w2v_model.wv[x[0]], filtered[:5]))
            )

            sentence_embedding = np.mean(arrlist, axis=0)

            if type(sentence_embedding) == np.ndarray:
                embeddings.append(sentence_embedding)
            else:
                embeddings.append(np.random.randn(embedding_size))
        return embeddings
