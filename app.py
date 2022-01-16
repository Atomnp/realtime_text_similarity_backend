from flask import Flask, request
from flask_restful import Resource, Api
from algorithms import get_similar
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)


class Similarity(Resource):
    def get(self):
        return {"hello": "world"}

    def post(self):
        json_data = request.get_json(force=True)
        qn = json_data["question"]
        similarity = get_similar(qn)
        x = {key: str(val) for key, val in similarity.items()}
        return x


api.add_resource(Similarity, "/")
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
