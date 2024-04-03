#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import time

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

    st.image('breast-cancer.jpg', caption="Breast Cancer Dataset")

    text = """he breast cancer dataset in scikit-learn is a well-known dataset used for binary 
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

    # Load the iris dataset
    data = load_breast_cancer()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

    st.subheader('Descriptive Statistics')
    st.write(df.describe(include='all').T)

    with st.expander('Click to browse the dataset'):
        st.write(df)

    with st.expander('Click to display unique values in each feature.'):
        # Get column names and unique values
        columns = df.columns
        unique_values = {col: df[col].unique() for col in columns}    
        
        # Display unique values for each column
        st.write("\n**Unique Values:**")
        for col, values in unique_values.items():
            st.write(f"- {col}: {', '.join(map(str, values))}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state
    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test


    # Preprocess the data (e.g., scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # store for later use
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.session_state.dataset_ready = True
    st.write('Dataset loadking complete.')

#run the app
if __name__ == "__main__":
    app()
