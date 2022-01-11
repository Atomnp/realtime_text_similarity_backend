import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

question_database = [
    "What is AVL Tree?",
    "Describe 4 principle of economics.",
    "Differentiate between process and threads",
    "Describe UML diagram and create one for online shopping system.",
    "Descrive VHDL in detail.",
    "Differentiate between dfs and bfs",
]


def embed(input):
    return model(input)


def create_dict(score, questions):
    return dict(zip(score, questions))


def get_similar(question):
    inp = [question] + question_database

    embeddings = embed([question] + question_database)
    distances = list(np.inner(embeddings, embeddings)[0][1:])
    similarity = create_dict(question_database, distances)
    return dict(sorted(similarity.items(), key=lambda x: x[1], reverse=True))


if __name__ == "main":
    print(get_similar("what is avl tree?"))
