from fileinput import filename
from search_index import AnnoyIndex
from encoders.arora import Arora
from encoders.use import UniversalEncoder
from encoders.bert import BERT
from encoders.word2vec import word2vec

filename = "/home/smloy/Downloads/dataset (2).txt"

if __name__ == "__main__":
    questions = []
    with open(filename, "r") as fp:
        questions = [x.strip().lower().split("?,") for x in fp.readlines() if x != "\n"]

    assert (
        len(questions[0]) == 2
    ), "first item should be sentence second item shoud be number representing unique question id"
    # encoder = UniversalEncoder(model_path="./USE")
    encoder = Arora()
    embeddings = encoder.encode_array([question[0]+"?" for question in questions])
    annoy_index = AnnoyIndex(dimension=len(embeddings[0]))
    print("questions ", questions)

    for question in questions:
        print(question)
        if not question[1]:
            print(question)

    annoy_index.build(embeddings, [question[1] for question in questions])
    accuracy = 0
        
    for question, unique_id in questions:
        no_of_similar_sent = [question[1] for question in questions].count(unique_id)
        neighbours = annoy_index.query(encoder.encode(question), k=no_of_similar_sent)
        print(neighbours)
        true_similar_count = neighbours.count(unique_id)
        accuracy += true_similar_count / no_of_similar_sent

    # for question, unique_id in questions:
    #     no_of_similar_sent = [question[1] for question in questions].count(unique_id)
    #     neighbours = annoy_index.query(encoder.encode(question), k=no_of_similar_sent)
    #     print(neighbours)
    #     denominator=0
    #     numerator=0
    #     for i in range(no_of_similar_sent):
    #         denominator+=1/(i+1)
    #         if neighbours[i]==unique_id:
    #             numerator += 1/(i+1)

    #     accuracy += numerator / denominator

    print(f"accuracy={accuracy/len(embeddings)*100}")

    # annoy_index.save("./indices/" + filename + str(self.current_algo) + ".ann")
    # self.index = annoy_index