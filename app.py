from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin

from algorithms import Algorithm, Runtime

from gensim.models.callbacks import CallbackAny2Vec

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

# initialize default runtime
runtime = Runtime()

# this callback is added in this file because word2vec is trainted using this callback
# and when that model is saved callback is saved too, and ite when we load the model
# model loader expects this callback to be in  __main__ module ie where the executaion starts
# so added here feel free to move it to arora.py if you find some way to add function to __main__ modeule from
# arora.py
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


class RequestHandler(Resource):
    def get(self):
        # print(request.args)
        new_algo = request.args["algo"]
        # try:
        runtime.switch_algo(algo=Algorithm[new_algo])
        # except Exception as e:
        #     print(e)
        #     return jsonify({"Error": "Cannot switch Algorithm"})

        return jsonify({"message": "Algorthm changed sucessfully"})

    def post(self):
        json_data = request.get_json(force=True)
        qn = json_data["question"]
        return runtime.get_similar(qn)

    # def put(self):
    #     """Handles request to swich algorithm used for similarity matching"""
    #     json_data = request.get_json(force=True)
    #     qn = json_data["question"]


api.add_resource(RequestHandler, "/")
if __name__ == "__main__":
    app.run(debug=True)
