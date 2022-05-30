import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
from tensorflow import keras

from functions import preprocess_b64, create_figure
import json

app = Flask(__name__)
model = keras.models.load_model('models/le_net5_v2.h5')
CORS(app, headers=['Content-Type'])


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/internals")
def internals():
    return render_template('internals.html')


@app.route("/models")
def models():
    return render_template('models.html')


@app.route('/hook2', methods=["GET", "POST", 'OPTIONS'])
def predict():
    """
    Decodes image, makes prediction, saves figure of prediction output, passes predictions to draw.js
    """
    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        img = preprocess_b64(image_b64)

        output = model.predict(img)
        preds = np.argsort(output)[0, ::-1]

        fig_data = create_figure(output)

        # Create the data sent to draw.js for output (top 3 predictions and bar chart of prediction certainties)
        output = {"pred" + str(i): str(preds[i]) for i in range(3)}
        output["fig_data"] = fig_data
    return json.dumps(output)


if __name__ == '__main__':
    app.run(debug=True)
