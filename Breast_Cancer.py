#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

    st.image('heart-disease.jpg', caption="Heart Disease Diagnosis using MLP-ANN")

    text = """
    This Streamlit app leverages various classifiers to predict the presence or absence of 
    heart disease based on the input of various heart symptoms. The data used to train 
    the model comes from the heart disease dataset 
    [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset].
    \nFactors such as chest pain, blood pressure, and cholesterol, 
    are predictors that can estimate the likelihood of having heart disease.
    \nDisclaimer: This app is for informational purposes only and should not be 
    used for definitive medical diagnosis. 
    Please consult a healthcare professional for any concerns about your heart health.
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

    text = """This data set dates from 1988 and consists of four databases: 
    Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, 
    including the predicted attribute, but all published experiments 
    refer to using a subset of 14 of them. The "target" variable refers to the 
    presence of heart disease in the patient. It is integer valued 0 = no 
    disease and 1 = disease.
    \nAttribute Information:

    age
    sex
    chest pain type (4 values)
    resting blood pressure
    serum cholestoral in mg/dl
    fasting blood sugar > 120 mg/dl
    resting electrocardiographic results (values 0,1,2)
    maximum heart rate achieved
    exercise induced angina
    oldpeak = ST depression induced by exercise relative to rest
    the slope of the peak exercise ST segment
    number of major vessels (0-3) colored by flourosopy
    thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
    """
    st.write(text)
    # Load the data dataset
    df = pd.read_csv('heart.csv', header=0)

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

    st.subheader('Descriptive Statistics')
    st.write(df.describe(include='all').T)

    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

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

    with st.expander('Click to show the graphs'):
        bins = [10, 30, 50, 70, 90]
        labels = ['10-29', '30-49', '50-69', '70-89']
        df['age_group'] = pd.cut(df['age'], bins, labels=labels)
        countplot(df, "age_group", "sex", "Age and Sex")
        countplot(df, "sex", "target", "Sex and Heart Disease")
        countplot(df, "age_group", "target", "Age Group and Heart Disease")
        countplot(df, "target", "fbs", "FBS>120 and Heart Disease")
        countplot(df, "target", "thal", "THAL and Heart Disease")
        plot_feature(df["trestbps"], df["chol"], 'trestbps', 'chol', 'trestbps VS chol')
        plot_feature(df["thalach"], df["chol"], 'thalach', 'chol', 'thalach VS chol')

def countplot(df, feature, grouping, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    # Create the countplot with clear title and legend
    p = sns.countplot(x=feature, data = df, hue=grouping,  palette='bright')
    ax.set_title(title, fontsize=14)

    # Display the plot
    plt.tight_layout()  # Prevent overlapping elements
    st.pyplot(fig)


def plot_feature(feature, target, labelx, labely, title):
    # Display the plots
    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter plot
    ax.scatter(feature, target)
    # Add labels and title
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_title(title)
    # Add grid
    ax.grid(True)
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
