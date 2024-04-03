#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import time

# Define the Streamlit app
def app():
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")

    st.subheader("Binary Classification of Breast Cancer with MLP Classifier")
    text = """The breast cancer dataset is a classic benchmark used in machine learning for binary classification. 
    In this case, the goal is to classify tumors as either malignant (cancerous) or benign (non-cancerous) 
    based on various features extracted from biopsies. 
    Load the breast cancer dataset using libraries like `scikit-learn` in Python.  This dataset typically 
    consists of features like mean radius, smoothness, texture, etc., and a target variable indicating the 
    class (malignant or benign).  Split the data into training and testing sets. The training set is used to 
    train the model, and the testing set is used to evaluate its performance.
    \n**MLP Classifier:** An MLP (Multi-Layer Perceptron) is a type of artificial neural network suitable for
     binary classification problems. It consists of an input layer (matching the number of features), 
     one or more hidden layers with activation functions (like sigmoid or ReLU), and an output layer 
     with a single neuron using a sigmoid activation function. The sigmoid function outputs a value 
     between 0 and 1, which can be interpreted as the probability of a sample belonging to the positive 
     class (malignant).
     \n**Training the Model:**
     * The training data is fed into the MLP. * The model adjusts the weights and biases between neurons 
     in each layer to minimize the difference between the predicted probabilities and the actual class 
     labels (malignant or benign) in the training data. * This process is called backpropagation and 
     uses an optimizer like stochastic gradient descent (SGD).
     \n**Evaluation:** * Once trained, the model is evaluated on the unseen testing data. Performance metrics
     like accuracy, precision, recall, and F1-score are used to assess how well the model distinguishes 
     between malignant and benign tumors.
     \n**Interpretation:** While MLPs are powerful, interpreting their inner workings can be challenging. 
     Techniques like feature importance analysis can help understand which features contribute most 
     to the model's predictions."""
    st.write(text)
    
   # Define MLP parameters    
    st.sidebar.subheader('Set the MLP Parameters')
    options = ["relu", "tanh", "logistic"]
    activation = st.sidebar.selectbox('Select the activation function:', options)

    options = ["lbfgs", "adam", "sgd"]
    solver = st.sidebar.selectbox('Select the solver:', options)

    hidden_layers = st.sidebar.slider(      
        label="How many hidden layers? :",
        min_value=5,
        max_value=250,
        value=10,  # Initial value
        step=5
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
        value=100,  
        step=10
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Define the MLP regressor model
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layers,5), 
            solver=solver, activation=activation, 
            max_iter=max_iter, random_state=42)

    text = """Recommended ANN parameters: solver=lbfgs, activation=relu, n_hidden_layer=150, max_iter=150"""
    st.write(text)

    if st.button('Start Training'):
        progress_bar = st.progress(0, text="Training the MLP regressor can take some time please wait...")

        # Train the model 
        clf.fit(X_train, y_train)

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Regressor training completed!") 

        st.subheader('Performance of the MLP-ANN Classifier on the Heart Disease Dataset')
        text = """We test the performance of the MLP Classifer using the 20% of the dataset that was
        set aside for testing. The confusion matrix and classification report are presented below."""
        st.write(text)

        # Make predictions on the test set
        y_test_pred = clf.predict(X_test)
        
        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("Performance test completed!") 
        
        st.subheader('Confusion Matrix')

        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

#run the app
if __name__ == "__main__":
    app()
