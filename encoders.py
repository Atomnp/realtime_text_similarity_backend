import abc  # for abstract base class
import tensorflow_hub as hub
import numpy as np


class Encoder(metaclass=abc.ABCMeta):  # python 3
    @abc.abstractmethod
    def encode(self, sentence: str):
        """required method"""
        return

    # @abc.abstractproperty
    # def some_property(self):
    #     """required property"""
    #     return


class UniversalEncoder(Encoder):
    def __init__(
        self, model_path: str = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ):
        self.model = hub.load(model_path)
        # pass

    @property
    def encode(self, sentence):
        embeddings = self.model([sentence])
