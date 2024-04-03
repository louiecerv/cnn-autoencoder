#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import time

# Define the Streamlit app
def app():
    if "dataset_ready" not in st.session_state:
        st.error("Dataset must be loaded. Click Heart Disease in the sidebar.")
        
    st.subheader("Binary Classification of Breast Cancer using KNN, SVM, and Naive Bayes")
    text = """This is a binary classification task aimed at predicting whether a tumor is malignant 
    (cancerous) or benign (non-cancerous) based on a set of features extracted from breast 
    tissue samples. 
    \n**Dataset:** A popular dataset for this task is the Wisconsin Diagnostic Breast Cancer 
    (WDBC) dataset available from the UCI Machine Learning Repository This dataset contains 
    information on several features like clump thickness, cell size uniformity, and
      marginal adhesion, along with a class label (malignant or benign) for each sample.
    \n**Algorithms:**
    \n* **K-Nearest Neighbors (KNN):** 
    * Classifies a new data point by considering the labels of its k nearest neighbors in the training data. 
    * In the breast cancer context, if the majority of the k nearest neighbors (based on feature similarity) are malignant in the training data, the new data point is classified as malignant and vice versa. 
    * Tuning the parameter k is crucial for optimal performance.
    \n* **Support Vector Machine (SVM):** 
    *  Finds a hyperplane in the feature space that best separates the data points belonging to different classes (malignant and benign) with the maximum margin. 
    *  This hyperplane can then be used to classify new data points. 
    *  SVMs are powerful for high-dimensional data and can handle non-linear relationships between features, but selecting the right kernel function can be important.
    \n* **Naive Bayes:**
    * A probabilistic classifier based on Bayes' theorem. 
    * It assumes independence between features, which may not always hold true for real-world data like breast cancer. 
    *  However, Naive Bayes is efficient to train and can be a good baseline for comparison. 
    *  It calculates the probability of a data point belonging to each class (malignant or benign) based on the individual feature probabilities. The class with the highest probability is assigned.
    \nThe performance of these algorithms can be evaluated using metrics like accuracy, precision, recall, 
    and F1-score. These metrics  consider the number of correctly classified and incorrectly classified 
    malignant and benign tumors. """
    st.write(text)

    #add the classifier selection to the sidebar
    clf = KNeighborsClassifier(n_neighbors=5)
    options = ['K Nearest Neighbor', 'Support Vector Machine', 'Naive Bayes']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Support Vector Machine':
        clf = SVC(kernel='linear')
        st.session_state['selected_model'] = 1
    elif selected_option=='Naive Bayes':        
        clf = GaussianNB()
        st.session_state['selected_model'] = 2
    else:
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 0

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    if st.button("Begin Training"):

        if selected_option =='K Nearest Neighbor':
            text = """KNN achieves good accuracy on the heart disease dataset, often 
            reaching around 85-90%. However, it can be slow for large datasets 
            due to needing to compare each test image to all training images. 
            Additionally, choosing the optimal number of neighbors (k) can be 
            crucial for performance."""
            classifier = 'K-Nearest Neighbor'
        elif st.session_state['selected_model'] == 1:   # SVM
            text = """SVM can also achieve high accuracy on this dataset, 
            similar to KNN. It offers advantages like being memory-efficient, 
            but choosing the right kernel function and its parameters 
            can be challenging."""
            classifier = 'Support Vector Machine'
        elif selected_option=='Naive Bayes': 
            text = """Naive Bayes is generally faster than the other two options but 
            may achieve slightly lower accuracy, typically around 80-85%. It performs 
            well when the features are independent, which might not perfectly hold true 
            for data found in the heart disease dataset."""
            classifier = "Naive Bayes"

        st.subheader('Performance of ' + classifier)
        st.write(text)

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))







#run the app
if __name__ == "__main__":
    app()