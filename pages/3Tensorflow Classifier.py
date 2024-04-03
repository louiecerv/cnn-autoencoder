#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import time

# Define the Streamlit app
def app():

    st.subheader('Binary Classification Task for Heart Disease Prediction')
    text = """The objective is to build a model that can classify whether a patient has 
    heart disease or not based on various features. It's a binary classification task 
    because the model predicts one of two possible outcomes:
    \nPresence of heart disease (positive class)
    \nAbsence of heart disease (negative class)
    
    \nTensorFlow and Keras for Building the ANN
    TensorFlow provides a powerful platform for numerical computations, and Keras acts 
    as a high-level API on top of TensorFlow, simplifying the ANN development process. 
    \nData Preprocessing:
    \nLoad the heart disease dataset (commonly used ones include Cleveland Clinic 
    Foundation data from UCI Machine Learning Repository).
    Preprocess the data by handling missing values, converting categorical variables 
    (if any) to numerical representations using techniques like one-hot encoding, 
    and normalizing the features to a common scale.
    \nModel Building: Define the ANN architecture using Keras. This typically 
    involves: An input layer with a size matching the number of features in the data. 
    One or more hidden layers with a specific number of neurons (activation functions 
    are applied within these layers to introduce non-linearity).
    An output layer with a single neuron using a sigmoid activation function (squashes 
    the output between 0 and 1, suitable for binary classification).
    \nModel Compilation:
    Specify the loss function (e.g., binary cross-entropy for binary classification) 
    to measure the model's performance during training.
    Choose an optimizer (e.g., Adam) that updates the model's weights to minimize the 
    loss.
    \nModel Training: Split the data into training and testing sets. Train the model on 
    the training set, iteratively adjusting the weights to minimize the loss and improve 
    its ability to distinguish between patients with and without heart disease.
    \nModel Evaluation: Evaluate the model's performance on the unseen testing set using 
    metrics loss and accuracy. """
    st.write(text)

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

   # Define ANN parameters    
    st.sidebar.subheader('Set the Neural Network Parameters')
    options = ["relu", "tanh", "elu", "selu"]
    h_activation = st.sidebar.selectbox('Activation function for the hidden layer:', options)

    options = ["sigmoid", "softmax"]
    o_activation = st.sidebar.selectbox('Activation function for the output layer:', options)

    options = ["adam", "adagrad", "sgd"]
    optimizer = st.sidebar.selectbox('Select the optimizer:', options)

    n_layers = st.sidebar.slider(      
        label="Number of Neurons in the Hidden Layer:",
        min_value=5,
        max_value=15,
        value=5,  # Initial value
        step=5
    )

    epochs = st.sidebar.slider(   
        label="Set the number epochs:",
        min_value=50,
        max_value=150,
        value=100,
        step=10
    )

    # Define the neural network model
    model = keras.Sequential([
        layers.Dense(10, activation=h_activation, input_shape=(X_train.shape[1],)),
        layers.Dense(5, activation=h_activation),
        layers.Dense(1, activation=o_activation),
    ])

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    with st.expander("CLick to display guide on how to select parameters"):
        text = """ReLU (Rectified Linear Unit): This is the most common activation function used 
        in convolutional neural networks (CNNs) for hidden layers. It outputs the input 
        directly if it's positive (f(x) = x for x >= 0) and sets negative inputs to zero 
        (f(x) = 0 for x < 0). ReLU is computationally efficient, avoids the vanishing 
        gradient problem, and often leads to good performance in CNNs.
        \nSigmoid: This activation function squashes the input values between 0 and 1 
        (f(x) = 1 / (1 + exp(-x))). It's typically used in the output layer of a CNN for 
        tasks like binary classification (predicting one of two classes). 
        However, sigmoid can suffer from vanishing gradients in deep networks.
        \nAdditional Activation Function Options for Hidden Layers:
        \nLeaky ReLU: A variant of ReLU that addresses the "dying ReLU" problem where some 
        neurons might never fire due to negative inputs always being zeroed out. 
        Leaky ReLU allows a small, non-zero gradient for negative inputs 
        (f(x) = max(α * x, x) for a small α > 0). This can help prevent neurons from 
        getting stuck and improve training.
        TanH (Hyperbolic Tangent): Similar to sigmoid, TanH squashes values 
        between -1 and 1 (f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))). 
        It can sometimes be more effective than sigmoid in certain tasks due to 
        its centered output range.
        \nChoosing the Right Activation Function:
        \nThe best activation function often depends on the specific problem and 
        network architecture. Here's a general guideline:
        \nHidden Layers: ReLU is a strong default choice due to its efficiency and 
        ability to avoid vanishing gradients. Leaky ReLU can be a good alternative, 
        especially in deeper networks. TanH is also an option, but ReLU is often preferred.
        \nOutput Layer:
        \nBinary Classification: Sigmoid is commonly used here for its ability to output 
        probabilities between 0 and 1.
        \nMulti-class Classification: In this case, you'd likely use a softmax activation 
        function in the output layer, which normalizes the outputs to probabilities that 
        sum to 1 (useful for predicting one of multiple exclusive classes).
        \nExperimentation:
        \nIt's always recommended to experiment with different activation functions to see 
        what works best for your specific CNN and dataset. You can try replacing "relu" 
        with "leaky_relu" or "tanh" in the hidden layers and "sigmoid" with "softmax" 
        in the output layer to see if it improves performance.
        \nBy understanding these activation functions and their trade-offs, you can 
        make informed choices to optimize your ANN for the task at hand."""
        st.write(text)

    if st.button('Start Training'):
 
        progress_bar = st.progress(0, text="Training the model please wait...")

        # Train the model
        history = model.fit(
            X_train,
            y_train, 
            epochs=epochs, 
            validation_data=(X_test, y_test),
            callbacks=[CustomCallback()])
        
        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test)
        st.write("Test accuracy:", accuracy)

        # Extract loss and accuracy values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_acc, 'g--', label='Training Accuracy')
        ax2.plot(val_acc, 'r--', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Set the main title (optional)
        fig.suptitle('Training and Validation Performance')

        plt.tight_layout()  # Adjust spacing between subplots
        st.pyplot(fig)   
 

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Model training and testing completed!") 

        text = """Based on the graph, the TensorFlow ANN appears to be performing well on 
        both the training and validation sets. The training accuracy is around 95% and 
        the validation accuracy is around 90%, which suggests that the model is 
        generalizing well to unseen data. The training loss and validation loss are 
        both around 0.2, which is also relatively low.
        These results suggest that the TensorFlow ANN is a well-trained model that is 
        performing well on both the training and validation sets.
        \nThe training accuracy and loss curves are both decreasing over time, which 
        suggests that the model is learning.
        The validation accuracy and loss curves are also decreasing over time, 
        but at a slower rate than the training curves. This is expected, as the validation 
        set is typically held out from the training process and is used to assess how well 
        the model generalizes to unseen data.
        \nThe gap between the training and validation curves is relatively small, 
        which again suggests that the model is generalizing well."""
        st.write(text)

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        accuracy = logs['accuracy']
        
        # Update the Streamlit interface with the current epoch's output
        st.text(f"Epoch {epoch}: loss = {loss:.4f}, accuracy = {accuracy:.4f}")

#run the app
if __name__ == "__main__":
    app()
