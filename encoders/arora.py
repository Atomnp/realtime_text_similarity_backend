from encoders.base import Encoder
import pickle

from fse import Vectors, IndexedList
from fse.models import uSIF,Average
# import gensim.downloader as api
# data = api.load("quora-duplicate-questions")
# wv = Vectors.from_pretrained("word2vec-google-news-300")

class Arora(Encoder):
    def __init__(self):
        self.wv=Vectors.from_pretrained("paranmt-300")
        self.model = Average(self.wv, workers=1, lang_freq="en")

    def encode(self, sentence: str, embedding_size=300, a=1e-3):
        sentence=sentence.replace("?","")
        iList = IndexedList([sentence.split()])
        self.model.train(iList)
        return self.model.sv[0]
        # return self.model.infer([tmp])
    
    def encode_array(self, sentences):
        iList = IndexedList([s.split() for s in sentences])
        self.model.train(iList)
        return self.model.sv.vectors

if __name__=='__main__':
    sentences_a = ["Hello there", "how are you?"]
    sentences_b = ["today is a good day", "Lorem ipsum"]


    from fse.models import uSIF
    model = uSIF(glove, workers=1, lang_freq="en")

    sentences = []
    j = 0
    for d in data:
        if j == 100:
            break
        j += 1
        for i in range(8):
            sentences.append(d["question1"].split())
            sentences.append(d["question2"].split())
    s = IndexedList(sentences)
    # print(sentences[:5])
    # print(len(s))

    model.train(s)
    # print(len(model.sv[0]), "lenght")

    print(model.sv)

    # print(s[100])
    # tmp = ("Hello my friends".split(), 0)
    # model.infer([tmp])