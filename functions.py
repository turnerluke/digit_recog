
import numpy as np
import cv2
import matplotlib.pyplot as plt



def pre_process_image(im):
    X = np.asarray(bytearray(im), dtype="uint8")
    X = cv2.imdecode(X, cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots()
    ax.imshow(X, cmap='Greys')
    plt.show()

    # Resizing and reshaping to keep the ratio.
    X = cv2.resize(X, (28, 28), interpolation=cv2.INTER_AREA)
    X = np.array(X).astype('float32')

    # Flatten & normalize X
    X = X.reshape((1, np.prod(X.shape)))
    X = X / np.max(X)

    return X