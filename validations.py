from fileinput import filename
from search_index import AnnoyIndex
from encoders.arora import Arora
from encoders.use import UniversalEncoder
from encoders.bert import BERT
from encoders.word2vec import word2vec

filename = "./uploads/dataset (2).txt"


def get_accuracy(encoder, sentences):

    embeddings = encoder.encode_array([question[0] + "?" for question in questions])
    annoy_index = AnnoyIndex(dimension=len(embeddings[0]))
    annoy_index.build(embeddings, [question[1] for question in questions])
    accuracy = 0

    for question, unique_id in questions:
        no_of_similar_sent = [question[1] for question in questions].count(unique_id)
        neighbours = annoy_index.query(encoder.encode(question), k=no_of_similar_sent)
        # print(neighbours)
        true_similar_count = neighbours.count(unique_id)
        accuracy += true_similar_count / no_of_similar_sent

    return accuracy / len(embeddings) * 100

def get_accuracy_mrr(encoder, sentences):

    embeddings = encoder.encode_array([question[0] + "?" for question in questions])
    annoy_index = AnnoyIndex(dimension=len(embeddings[0]))
    annoy_index.build(embeddings, [question[1] for question in questions])
    accuracy = 0

    for question, unique_id in questions:
        no_of_similar_sent = [question[1] for question in questions].count(unique_id)
        neighbours = annoy_index.query(encoder.encode(question), k=no_of_similar_sent)
        # print(neighbours)
        denominator=0
        numerator=0
        for i in range(no_of_similar_sent):
            denominator+=1/(i+1)
            if neighbours[i]==unique_id:
                numerator += 1/(i+1)

        accuracy += numerator / denominator

    return accuracy/len(embeddings)*100

from gensim.models.callbacks import CallbackAny2Vec


def identity_tokenizer(text):
    return text


class callback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


if __name__ == "__main__":
    questions = []
    with open(filename, "r") as fp:
        questions = [x.strip().lower().split("?,") for x in fp.readlines() if x != "\n"]

    assert (
        len(questions[0]) == 2
    ), "first item should be sentence second item shoud be number representing unique question id"
    # encoder = UniversalEncoder(model_path="./USE")
    question_array = [question[0] + "?" for question in questions]

    bert_accuracy = get_accuracy(BERT(), question_array)
    arora_accuracy = get_accuracy(Arora(), question_array)
    use_accuracy = get_accuracy(UniversalEncoder(), question_array)
    word2vec_accuracy = get_accuracy(word2vec(), question_array)
    print(
        "bert_accuracy: {0}\n arora_accuracy: {1}\n use_accuracy: {2}\n word2vec_accuracy: {3}".format(
            bert_accuracy, arora_accuracy, use_accuracy, word2vec_accuracy
        )
    )

    bert_accuracy = get_accuracy_mrr(BERT(), question_array)
    arora_accuracy = get_accuracy_mrr(Arora(), question_array)
    use_accuracy = get_accuracy_mrr(UniversalEncoder(), question_array)
    word2vec_accuracy = get_accuracy_mrr(word2vec(), question_array)
    print(
        "bert_accuracy: {0}\n arora_accuracy: {1}\n use_accuracy: {2}\n word2vec_accuracy: {3}".format(
            bert_accuracy, arora_accuracy, use_accuracy, word2vec_accuracy
        )
    )
