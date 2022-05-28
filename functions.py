import base64
import numpy as np
import cv2
from PIL import Image
import io
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_b64(b64):
    """
    Preprocesses a base 64 image, returns numpy array ready to pass to model

    """

    image_encoded = b64.split(',')[1]
    image = base64.decodebytes(image_encoded.encode('utf-8'))

    img = Image.open(io.BytesIO(image))
    img = np.array(img)
    img = np.mean(img, axis=2)  # To B&W image
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)  # To (32, 32) for model input

    # 1) 0 -> 255
    img[img == 0] = 255
    # 2) Round numbers (canvas returns 254.999 for some reason)
    img = img.round()
    # 3) Scale vals from 0 -> 1
    img = (img - img.min()) / (img.max() - img.min())
    # 4) Inverse vals (vals = 1 - vals)
    img = 1 - img
    # 5) Standard scale vals (vals = (vals - vals_mean) / std_vals
    img = (img - img.mean()) / img.std()

    img = np.expand_dims(np.expand_dims(img, 0), 3)  # To (None, 32, 32, None)
    return img

def create_figure(output):
    """
    Takes the model output and returns a base64 encoded image of a barchart of the model output.

    """
    preds = np.argsort(output)[0, ::-1]
    # Plot the model output
    # Get a color palette, with larger values being darker
    palette = sns.color_palette("Blues_d", len(preds))
    rank = preds.argsort().argsort()  # http://stackoverflow.com/a/6266510/1628638
    palette = [palette[r] for r in rank]
    # Make barplot of the certainties
    fig, ax = plt.subplots()
    sns.barplot(x=np.arange(10), y=output[0, :] * 100, ax=ax, palette=palette)
    ax.set_ylabel("Confidence (%)")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return fig_data