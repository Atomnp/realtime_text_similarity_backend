import annoy
import pickle


class AnnoyIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = annoy.AnnoyIndex(self.dimension)

    def query(self, vector, k=10):
        indices = self.index.get_nns_by_vector(list(vector), k)
        return [self.labels[i] for i in indices]

    def load(self, path):
        label_path = path.split(".")[0] + ".labels"
        self.index = annoy.AnnoyIndex(self.dimension)
        with open(label_path, "rb") as fp:
            self.labels = pickle.load(fp)
        self.index.load(path)
