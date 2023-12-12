import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def preprocess_image(img: np.ndarray) -> np.ndarray:

    img = np.mean(img, axis=2)  # To B&W image (2 dimensional)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)  # To (32, 32) for model input

    # Scale vals from 0 -> 1
    img = (img - img.min()) / (img.max() - img.min())
    # Inverse vals (vals = 1 - vals)
    img = 1 - img
    # Standard scale vals (vals = (vals - vals_mean) / std_vals
    img = (img - img.mean()) / img.std()

    img = np.expand_dims(np.expand_dims(img, 0), 3)  # Reshape to (None, 32, 32, None)
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

    # Get the figure as a numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    figure_array = np.array(canvas.renderer.buffer_rgba())

    return figure_array