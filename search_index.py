import annoy
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class AnnoyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = annoy.AnnoyIndex(self.dimension)

    def query(self, vector, k=10):
        indices = self.index.get_nns_by_vector(list(vector), k)
        return [self.labels[i] for i in indices]

    def query_cosine(self, vector, k=10):
        indices = self.index.get_nns_by_vector(list(vector), k)
        return [
            (
                self.labels[i],
                cosine_similarity([vector], [self.index.get_item_vector(i)])[0][0],
            )
            for i in indices
        ]

    def load(self, path):
        assert path[-4:] == ".ann", print("Path must be given to some .ann file")
        label_path = path[:-4] + ".labels"
        self.index = annoy.AnnoyIndex(self.dimension)
        with open(label_path, "rb") as fp:
            self.labels = pickle.load(fp)
        self.index.load(path)

    def build(self, vectors, labels, number_of_trees=5):
        self.vectors = vectors
        self.labels = labels

        for i, vec in enumerate(vectors):
            if not np.isnan(np.sum(vec)):
                self.index.add_item(i, vec)
        self.index.build(number_of_trees)

    def save(self, path):
        label_path = path.split(".ann")[0] + ".labels"
        print(label_path)
        with open(label_path, "wb") as fp:
            pickle.dump(self.labels, fp)
        self.index.save(path)
