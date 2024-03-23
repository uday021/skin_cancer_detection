from flask import Flask
from predict import classify_using_bytes
from werkzeug.datastructures import FileStorage
from flask_restx import Api, Namespace, Resource, reqparse

predict_args = reqparse.RequestParser()
predict_args.add_argument(name = "image", type=FileStorage, location="files", required=True, help="This filed is required")

predict_controller = Namespace(
    name="predict",
    description="Upload image and get prediction",
    path="/predict"
)

api = Api(
    title="Skin Cancer Prediction",
    description="get all details about the skin disease",
    version="1.0",
    validate=True,
    doc="/"
)

@predict_controller.route("/skin-cancer")
class PredictResource(Resource):

    @predict_controller.expect(predict_args)
    def post(self):
        args = predict_args.parse_args()

        image = args['image']
        result = classify_using_bytes(image.read(), "weights.h5", 28)

        return result

api.add_namespace(predict_controller)

app = Flask(__name__)
api.init_app(app)

if __name__ == "__main__":
    app.run(debug=True)
