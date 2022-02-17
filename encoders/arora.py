# from gensim.models.word2vec import Word2Vec
from gensim.models import Word2Vec
from encoders.base import Encoder
import numpy as np

# from sklearn.decomposition import PCA
import pickle

from gensim.models.callbacks import CallbackAny2Vec


class Arora(Encoder):
    def __init__(self):
        self.w2v_model = Word2Vec.load("./indices/word2vec_6_iter.model")
        self.pca0 = np.load("./indices/pca0_arora.npy", allow_pickle=True)
        with open("./indices/wf_arora.pickle", "rb") as fp:
            self.wf = pickle.load(fp)
            self.unique_words = sum(self.wf.values())

    def get_word_frequency(self, word_text):
        return self.wf[word_text] / self.unique_words

    def encode(self, sentence: str, embedding_size=100, a=1e-3):
        vs = np.zeros(embedding_size)
        for word in sentence:
            a_value = a / (a + self.get_word_frequency(word))
            if word in self.w2v_model.wv:
                vs = np.add(vs, np.multiply(a_value, self.w2v_model.wv[word]))
        vs = np.divide(vs, 1 if not len(sentence) else len(sentence))

        # calculate PCA of this sentence
        # pca.transform([vs])
        u = self.pca0  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        return np.subtract(vs, np.multiply(u, vs))
