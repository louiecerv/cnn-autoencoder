#Input the relevant libraries

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


    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []


    if "dataset_ready" not in st.session_state:
        st.session_state.dataset_ready = False 

    text = """Three-way comparison of ML Classifiers, MLP and Tensorflow Artificial Neural Networks on the Heart Disease Dataset"""
    st.header(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    #st.image('breast-cancer.jpg', caption="Breast Cancer Dataset")

    text = """The breast cancer dataset in scikit-learn is a well-known dataset used for binary 
        classification tasks. It contains data collected from patients diagnosed with breast cancer. 
        Here's a breakdown of the dataset:
        \nSource: The data consists of features extracted from digitized images of fine needle 
        aspirates (FNA) of breast masses.
        \nFeatures: The dataset includes 569 data points, each with 30 numerical features. 
        These features represent various characteristics of the cell nuclei, such as radius, 
        texture, perimeter, and area.
        \nTarget: The target variable indicates the class, whether the tumor is malignant 
        (cancerous) or benign (non-cancerous). There are 212 malignant cases and 357 benign cases
    """
    st.write(text)

    text = """
    An MLP (Multi-Layer Perceptron) classifier can be used to analyze a heart disease dataset and 
    predict whether a patient has heart disease or not. Here's how it works in this context:
    \nBinary Classification:
    The MLP aims for binary classification, meaning the output will be either 0 (no heart disease) or 1 (heart disease).
    \nData Preprocessing:
    The heart disease dataset contain various features like age, blood pressure, cholesterol levels, etc. 
    These features might need scaling or normalization for the MLP to process them efficiently.
    
    \nMLP Architecture:
    The MLP is a type of artificial neural network with an interconnected layer structure.
    In this case, the input layer will have the size matching the number of features in the heart 
    disease data (e.g., age, blood pressure). There will be one or more hidden layers containing 
    a number of artificial neurons. These hidden layers extract complex patterns from the data.  
    The final output layer will have a single neuron with a sigmoid activation function. 
    This neuron outputs a value between 0 and 1, which is interpreted as the probability of 
    having heart disease (closer to 1) or not (closer to 0).

    \nTraining:
    The MLP is trained using a labeled dataset where each data point has a confirmed classification 
    (heart disease or no disease) associated with its features.
    During training, the MLP adjusts the weights and biases between its artificial neurons to 
    minimize the error between its predicted probabilities and the actual labels in the training data.
    A common training algorithm for MLPs is backpropagation, which calculates the error and propagates 
    it backward through the network to update the weights and biases.
    \nPrediction:
    Once trained, the MLP can predict the probability of heart disease for new, unseen data points 
    based on their features. A threshold is typically set on the output probability (e.g., 0.5). 
    Values above the threshold are classified as having heart disease, while those below are 
    classified as healthy."""

    st.write(text)

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
    path = "landscape Images\gray"
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

    model = model()
    model.summary()

    if st.button("Start"):
         
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mean_absolute_error',
              metrics = ['acc'])
        
        model.fit(train_g, train_c, epochs = 50,batch_size = 50,verbose = 0)

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


def model():
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

#run the app
if __name__ == "__main__":
    app()
