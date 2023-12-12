import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

from utils import preprocess_image, create_figure

model = keras.models.load_model('models/le_net5_v2.h5')


def main():
    PAGES = {
        "App": app_page,
        "Model": model_page,
        "Streamlit App": streamlit_app_page,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/turnerluke">@turnerluke</a></h6>',
            unsafe_allow_html=True,
        )


def app_page():
    drawing_mode = "freedraw"
    stroke_width = 20

    stroke_color = '#000000'
    bg_color = "#eee"
    realtime_update = True

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="full_app",
    )

    image_data: np.ndarray = canvas_result.image_data

    if image_data is not None:
        mean_values = image_data.mean(axis=2)

        if not np.all(np.isclose(mean_values, mean_values[0, 0])):
            img = preprocess_image(image_data)

            output = model.predict(img)
            preds = np.argsort(output)[0, ::-1]

            # Display the prediction
            st.header(f"Prediction: {preds[0]}")

            # Create certainties bar chart
            figure = create_figure(output)
            st.subheader("Model Certainties:")
            st.image(figure)


def model_page():
    st.markdown(
        """
        # Model
        
        The model used is the LeNet-5 v2.0 convolutional neural network (CNN), 
        originally presented in 
        [LeCun et al., Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).
        
        The model was implemented in Keras as shown here:
        
        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/turnerluke/digit_recog/blob/main/models/LeNet_5_train.ipynb)
        """
    )

    st.markdown(
        """
        # Convolutional Neural Network Structure
        
        | Layer (Type)                    | Output Shape       | Parameters |
        | ------------------------------  | ------------------ | ---------- |
        | Convolution_1 (conv2d)          | (None, 28, 28, 32) | 832        |
        | Convolution_2 (conv2d)          | (None, 24, 24, 32) | 25600      |
        | Batchnorm_1 (batchnormalization)| (None, 24, 24, 32) | 128        |
        | Activation_25 (activation)      | (None, 24, 24, 32) | 0          |
        | Max_pool_1 (maxpooling2d)       | (None, 12, 12, 32) | 0          |
        | Dropout_1 (dropout)             | (None, 12, 12, 32) | 0          |
        | Convolution_3 (conv2d)          | (None, 10, 10, 64) | 18496      |
        | Convolution_4 (conv2d)          | (None, 8, 8, 64 )  | 36864      |
        | Batchnorm_2 (batchnormalization)| (None, 8, 8, 64)   | 256        |
        | Activation_26 (activation)      | (None, 8, 8, 64)   | 0          |
        | Max_pool_2 (maxpooling2d)       | (None, 4, 4, 64)   | 0          |
        | Dropout_2 (dropout)             | (None, 4, 4, 64)   | 0          |
        | Flatten (flatten)               | (None, 1024)       | 0          |
        | Fully_connected_1 (dense)       | (None, 256)        | 262144     |
        | Batchnorm_3 (batchnormalization)| (None, 256)        | 1024       |
        | Activation_27 (activation)      | (None, 256)        | 0          |
        | Fully_connected_2 (dense)       | (None, 128)        | 32768      |
        | Batchnorm_4 (batchnormalization)| (None, 128)        | 512        |
        | Activation_28 (activation)      | (None, 128)        | 0          |
        | Fully_connected_3 (dense)       | (None, 84)         | 10752      |
        | Batchnorm_5 (batchnormalization)| (None, 84)         | 336        |
        | Activation_29 (activation)      | (None, 84)         | 0          |
        | Dropout_3 (dropout)             | (None, 84)         | 0          |
        | Output (dense)                  | (None, 10)         | 850        |
        
        
        """
    )

    st.image('images/LeNet 5 v2 Vis.png')
    st.markdown('**Figure 1:** LeNet-5 CNN Structure (Created with [NN-SVG](http://alexlenail.me/NN-SVG/LeNet.html))')

    st.markdown(
        """
        # Data
        
        The Modified National Institute of Standards and Technology (MNIST) dataset of handwritten digits was used as 
        the training and validation datasets for this model. The dataset was obtained from the `keras.datasets` module 
        as follows:
            
        ```python
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        ```
            
        This dataset consists of 60,000 training and 10,000 testing digits and their corresponding labels.

        The data is normalized to have a mean value of 0 and standard deviation of 1.

        Data augmentation is performed to prevent overfitting of the dataset, using the following random operations:
        - 10 degree rotations
        - 10% zoom
        - 10% horizontal shifts
        - 10% vertical shifts
        
        """
    )

    st.markdown(
        """
        # Training
        
        The model was trained for 30 epochs, as originally performed by LeCun et al. The loss function was categorical 
        cross entropy, and the accuracy metric was utilized to quantify performance.
        """
    )

    st.markdown(
        """
        # Performance
        
        This model achieved a final accuracy of 99.63% on the test set, after 30 epochs of training.
        
        """
    )
    st.image('images/training.png')
    st.markdown('**Figure 2:** Training and validation loss and accuracy versus training epoch')

    st.image('images/cm.png')
    st.markdown('**Figure 3:** Confusion matrix for the model on the MNIST dataset')

    st.markdown(
        """
        # Debugging
        
        If your digit sketches are not being properly predicted by the model, try drawing them to take up a majority of 
        the space on the canvas. Please note, that LeNet-5 v2.0 is exceptional at recognizing the MNIST dataset, so to 
        ensure a similar performance on your sketches, they should closely match the data.
        """
    )
    st.image('images/digits.png')
    st.markdown('**Figure 4:** Example of digits that are recognized well by the model')


def streamlit_app_page():
    st.markdown(
        """
        # Streamlit App
        
        This project's deployment originally started as a Flask app, deployed to heroku. After Heroku ended their free 
        tier, it was refactored to be deployed on Render. However, Render's free tier stopped allowing ML models to be
        deployed, so it was refactored again to be deployed on Streamlit Sharing.
        
        The original Flask app was a much more complex deployment, requiring JavaScript, HTML, and CSS. Streamlit, on
        the other hand, is a much simpler workflow for Data Scientists to deploy their apps.
        
        The GitHub repository for this project can be found [here](https://github.com/turnerluke/digit_recog).
        
        """
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Drawn Digit Recognition",
        page_icon="images/favicon.png",
    )
    st.title("Drawn Digit Recognition")
    main()
