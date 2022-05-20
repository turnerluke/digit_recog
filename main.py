
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import base64
from tensorflow import keras
from PIL import Image
import io
import cv2

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
	Decodes image and uses it to make prediction.
	"""
    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))

        img = Image.open(io.BytesIO(image))
        img = np.array(img)
        img = np.mean(img, axis=2)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        #img.reshape(1, 32, 32, 1)
        img = np.expand_dims(np.expand_dims(img, 0), 3)

        print(img.shape)

        #plt.imshow(img)
        #plt.show()

        prediction = model.predict(img)
        prediction = {
            'answer': str(np.argmax(prediction)),
            'small_predictions': ''.join(str([p for p in prediction]))
        }
        # TODO: THis output needs to match the original

    return json.dumps(prediction)


if __name__ == '__main__':
    app.run(debug=True)