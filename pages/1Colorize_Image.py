
import streamlit as st
import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import time
from keras import layers
import contextlib
import io  # Import the io module
import random

# Suppress the oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Handle the deprecated tf.reset_default_graph warning (if using TensorFlow 2)
if tf.version.VERSION.startswith('2'):
    from tensorflow.compat.v1 import reset_default_graph  # Use compat.v1 for deprecated function
else:
    reset_default_graph = tf.compat.v1.reset_default_graph  # For TensorFlow 1 compatibility

# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)

def plot_images(color, grayscale):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))  # Create figure and subplots

    ax1.imshow(color)  # Display color image on first subplot
    ax1.set_title('Color Image', color='green', fontsize=20)  # Set title

    ax2.imshow(grayscale)  # Display grayscale image on second subplot
    ax2.set_title('Grayscale Image', color='black', fontsize=20)  # Set title

    st.pyplot(fig)

# Define the Streamlit app
def app():

    st.subheader("How to use this Data App")

    text = """This data app helps you train and test a model on your images. To ensure everything 
    runs smoothly, follow these steps in the correct order:
    \n1. **Load Your Images:**  Start by uploading your image data. The app will accept various image 
    formats, allowing you to easily import your dataset.
    \n2. **Initialize the Model:** Once your images are loaded, the app will guide you through selecting 
    and customizing a model architecture. This is where you define the "brain" of your system that will 
    learn from your data.
    \n3. **Train and Test:**  With both images and model ready, you can initiate the training process. 
    The app will split your data and train the model on one part while evaluating its performance on the other. 
    This helps ensure your model generalizes well to unseen data.
    These steps build upon each other. Uploading your data first allows the app to understand the type of 
    information your model needs to learn from. Then, with the data loaded, you can define the model architecture 
    to process that information effectively. Finally, after everything is set up, you can train and test your 
    model to see how well it performs."""
    st.write(text)

   # Define CNN parameters    
    st.sidebar.subheader('Set the CNN Parameters')
    options = ["relu", "leaky_relu", "tanh", "elu", "selu"]
    h_activation = st.sidebar.selectbox('Activation function for the hidden layer:', options)

    options = ["sigmoid", "softmax"]
    o_activation = st.sidebar.selectbox('Activation function for the output layer:', options)

    n_neurons = st.sidebar.slider(      
        label="Number of Neurons in the Hidden Layer:",
        min_value=32,
        max_value=1024,
        value=512,  # Initial value
        step=32
    )

    epochs = st.sidebar.slider(   
        label="Set the number epochs:",
        min_value=4,
        max_value=100,
        value=4,
        step=1
    )    

    batch_size = st.sidebar.slider(   
        label="Set the batch size:",
        min_value=16,
        max_value=64,
        value=32,
        step=16
    )      

    if st.sidebar.button("Load Images"):
        progress_bar = st.progress(0, text="Loading the images, please wait...")

        # defining the size of the image
        SIZE = 80
        color_img = []
        path = 'landscape Images/color'
        files = os.listdir(path)
        files = sorted_alphanumeric(files)
        for i in tqdm(files):    
            if i == '6000.jpg':
                break
            else:    
                img = cv2.imread(path + '/'+i,1)
                # open cv reads images in BGR format so we have to convert it to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #resizing image
                img = cv2.resize(img, (SIZE, SIZE))
                img = img.astype('float32') / 255.0
                color_img.append(img_to_array(img))

        gray_img = []
        path = "landscape Images/gray"
        files = os.listdir(path)
        files = sorted_alphanumeric(files)
        for i in tqdm(files):
            if i == '6000.jpg':
                break
            else: 
                img = cv2.imread(path + '/'+i,1)

            #resizing image
            img = cv2.resize(img, (SIZE, SIZE))
            img = img.astype('float32') / 255.0
            gray_img.append(img_to_array(img))
        
        for i in range(3,10):
            plot_images(color_img[i],gray_img[i])

        train_gray_image = gray_img[:5500]
        train_color_image = color_img[:5500]

        test_gray_image = gray_img[5500:]
        test_color_image = color_img[5500:]

        # reshaping
        train_g = np.reshape(train_gray_image,(len(train_gray_image),SIZE,SIZE,3))
        train_c = np.reshape(train_color_image, (len(train_color_image),SIZE,SIZE,3))
        # save variables to session

        st.write('Train color image shape:',train_c.shape)

        test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,3))
        test_color_image = np.reshape(test_color_image, (len(test_color_image),SIZE,SIZE,3))
        st.write('Test color image shape',test_color_image.shape)
    
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Image dataset loading completed!") 

        model = get_model()

        # Capture the summary output
        st.write("Model summary details:")
        with contextlib.redirect_stdout(io.StringIO()) as new_stdout:            
            model.summary()
            summary_str = new_stdout.getvalue()
        # Display the summary using st.text()
        st.text(summary_str)

        progress_bar = st.progress(0, text="Training the model, please wait...")

        # Hyperparameter adjustments
        learning_rate = 0.001  # Experiment with different values

        # Compile the model with optimized parameters
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_absolute_error',  # Consider using MAE for images
            metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
        )

        # Train the model with adjustments for efficiency and early stopping
        history = model.fit(
            train_g, train_c,
            epochs=epochs,  # Adjust based on validation performance
            batch_size=batch_size,  # Adjust based on GPU memory
            verbose=1,
            validation_data=(test_gray_image, test_color_image),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                CustomCallback(),
                # Consider adding learning rate scheduler (e.g., ReduceLROnPlateau)
            ]
        )

         # Extract loss and MAE/MSE values from history
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_mae = history.history['mean_absolute_error']
        val_mae = history.history['val_mean_absolute_error']
        train_mse = history.history['mean_squared_error']
        val_mse = history.history['val_mean_squared_error']

        # Create the figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize for better visualization

        # Plot loss on the first subplot (ax1)
        ax1.plot(train_loss, label='Training Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy on the second subplot (ax2)
        ax2.plot(train_mae, 'g--', label='Training Mean Absolute Error')
        ax2.plot(train_mse, 'g--', label='Training Mean Squared Error')
        ax2.plot(val_mae, 'r--', label='Validation Mean Absolute Error')
        ax2.plot(val_mse, 'r--', label='Validation Mean Squared Error')
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
        st.success("Model training completed!")         

        start_img  = random.randint(0, 400)
        end_img = start_img + 8
        for i in range(start_img, end_img):
            predicted = np.clip(model.predict(test_gray_image[i].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
            plot_3images(test_color_image[i], test_gray_image[i], predicted)

def down(filters, kernel_size):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2))
    downsample.add(layers.LeakyReLU())
    downsample.add(layers.BatchNormalization())  # Added batch normalization
    return downsample

def up(filters, kernel_size):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2))
    upsample.add(layers.LeakyReLU())
    upsample.add(layers.BatchNormalization())  # Added batch normalization
    return upsample

def get_model():
    inputs = layers.Input(shape=[80, 80, 3])

    # Encoder
    d1 = down(64, (3, 3))(inputs)
    d2 = down(128, (3, 3))(d1)
    d3 = down(256, (3, 3))(d2)  # Reintroduced removed layer

    # Bottleneck
    bottleneck = layers.Conv2D(64, (3, 3), padding='same')(d3)  # Strengthened bottleneck

    # Decoder
    u1 = up(128, (3, 3))(bottleneck)
    u1 = layers.concatenate([u1, d2])
    u2 = up(64, (3, 3))(u1)  # Reintroduced upsampling layer
    u2 = layers.concatenate([u2, d1])
    u3 = up(3, (3, 3))(u2)

    # Output
    outputs = layers.Conv2D(3, (3, 3), strides=1, padding='same', activation='tanh')(u3)  # Using tanh

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# defining function to plot images pair
def plot_3images(color, grayscale, predicted):
    fig, axes = plt.subplots(1, 3, figsize=(15, 15))  # Create a figure with 3 subplots

    # Set titles for each subplot
    axes[0].set_title('Color Image', color='green', fontsize=20)
    axes[1].set_title('Grayscale Image', color='black', fontsize=20)
    axes[2].set_title('Predicted Image', color='red', fontsize=20)

    # Display images on each subplot
    axes[0].imshow(color)
    axes[1].imshow(grayscale)
    axes[2].imshow(predicted)
    st.pyplot(fig)

# Define a custom callback function to update the Streamlit interface
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get the current loss and accuracy metrics
        loss = logs['loss']
        mse = logs['mean_squared_error']
        
        # Update the Streamlit interface with the current epoch's output
        st.text(f"Epoch {epoch}: loss = {loss:.4f} Mean Squared Errror = {mse:.4f}")

#run the app
if __name__ == "__main__":
    app()
