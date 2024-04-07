
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

    text = """##How to Use this Data App
    This data app helps you train and test a model on your images. To ensure everything runs smoothly, 
    follow these steps in the correct order:
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

    if st.button("Load Images"):
        progress_bar = st.progress(0, text="Loading the images, please wait...")

        # defining the size of the image
        SIZE = 160
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
        st.write('Train color image shape:',train_c.shape)

        test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,3))
        test_color_image = np.reshape(test_color_image, (len(test_color_image),SIZE,SIZE,3))
        print('Test color image shape',test_color_image.shape)
    
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Image dataset loading completed!") 


    if st.button("Initialize Model"):

        model = get_model()
        model.summary()

    if st.button("Start Training"):
         
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',
              metrics = ['acc'])
        
        model.fit(train_g, train_c, epochs = 50, batch_size = 50,verbose = 0)

        model.evaluate(test_gray_image, test_color_image)

        for i in range(50,58):
            predicted = np.clip(model.predict(test_gray_image[i].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)
            plot_3images(test_color_image[i], test_gray_image[i], predicted)

def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample

def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample

def get_model():
    inputs = layers.Input(shape= [160,160,3])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    
    d5 = down(512,(3,3),True)(d4)
    #upsampling
    u1 = up(512,(3,3),False)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)

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

#run the app
if __name__ == "__main__":
    app()
