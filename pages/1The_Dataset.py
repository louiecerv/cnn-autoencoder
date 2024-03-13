#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import time

# Define the Streamlit app
def app():

    text = """This data set dates from 1988 and consists of four databases: 
    Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, 
    including the predicted attribute, but all published experiments 
    refer to using a subset of 14 of them. The "target" field refers to the 
    presence of heart disease in the patient. It is integer valued 0 = no 
    disease and 1 = disease.
    Attribute Information:
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

    st.write('Browse the Dataset')
    st.write(df)

     # Get column names and unique values
    columns = df.columns
    unique_values = {col: df[col].unique() for col in columns}    
    
    # Display unique values for each column
    st.write("\n**Unique Values:**")
    for col, values in unique_values.items():
        st.write(f"- {col}: {', '.join(map(str, values))}")

    st.write('Descriptive Statistics')
    st.write(df.describe().T)

    # Separate features and target variable
    X = df.drop('target', axis=1)  # Target variable column name
    y = df['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # store for later use
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    # Standardize features using StandardScaler (recommended)
    scaler = st.session_state["scaler"] 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.session_state.X_test_scaled = X_test_scaled

    # save the scaler object for later use
    st.session_state["scaler"] = scaler

   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["adam", "lbfgs", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
        value=10,  # Initial value
    )

    alpha = st.sidebar.slider(   
        label="Set the alpha:",
        min_value=.001,
        max_value=1.0,
        value=0.1,  # In1.0itial value
    )

    max_iter = st.sidebar.slider(   
        label="Set the max iterations:",
        min_value=100,
        max_value=300,
        value=120,  
        step=10
    )

    # Define the MLP regressor model
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layers,), 
            solver=solver, activation=activation, 
            max_iter=max_iter, random_state=42)

    #store the clf object for later use
    st.session_state.clf = clf

    if st.button("Show Graphs"):
        st.write("TO DO: insert some data plot here")

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take up to five minutes please wait...")

        # Train the model 
        train_model(X_train_scaled, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 
        st.write("Use the sidebar to open the Performance page.")
        
def train_model(X_train_scaled, y_train):
    clf = st.session_state.clf 
    clf.fit(X_train_scaled, y_train)
    st.session_state.clf = clf

#run the app
if __name__ == "__main__":
    app()
