from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

# initialize default runtime


class RequestHandler(Resource):
    def get(self):
        return jsonify({"message": "Algorthm changed sucessfully"})

    def post(self):
        json_data = request.get_json(force=True)
        qn = json_data["question"]
        dummy_list = [
            ["What is the origin of Nepali Bahuns?", 0.92],
            ["Are Nepali Bahuns native to Nepal?", 0.92],
            ["With the current resources can Nepal extend its territory?", 0.92],
            [
                "Why are so many Quora users posting questions that are readily answered on Google?",
                0.92,
            ],
            [
                "Why do people ask Quora questions which can be answered easily by Google?,0.92"
            ],
            ["Why should I use DuckDuckGo instead of Google?", 0.92],
            ["Why use Quora over Google, for factual answers?", 0.92],
            ["Why do people use Quora? Why don't they Google their answers?", 0.92],
        ]
        return dummy_list


# need to validate for file types or empty files (edge case)
class FileUploadHandler(Resource):
    def post(self):
        file = request.files["file"]
        file.save("uploads/" + file.filename)
        print(request.files["file"])
        # runtime.change_index("uploads/" + file.filename)
        return ["ok"]


api.add_resource(FileUploadHandler, "/files")

api.add_resource(RequestHandler, "/")
if __name__ == "__main__":
    app.run(debug=True)
