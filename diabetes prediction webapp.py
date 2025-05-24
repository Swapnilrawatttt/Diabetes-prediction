# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:24:31 2025

@author: swapn
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("D:/Model deployement/diabetes_model.sav", 'rb'))


#creating fxn for prediction
def diabetes_prediction(input_data):
    # Change input data to numpy array
    input_data = np.asarray(input_data)

    # Reshape the array for predicting one instance
    input_data_reshaped = input_data.reshape(1, -1)

    # Standardize the input data
    #13std_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Output result
    if prediction[0] == 0:
        return "No Diabetes"
    else:
        return "The person is Diabetic"
    
def main():
    #giving a title
    st.title("Diabetes prediction Webapp")
    
    Pregnancies =st.text_input("Enter No. Pregnancies: ")
    Glucose = st.text_input("Enter Glucose Level: ")
    BloodPressure = st.text_input("Enter Blood Pressure level: ")
    SkinThickness = st.text_input("Enter Skin Thickness: ")
    Insulin = st.text_input("Enter Insulin Level: ")
    BMI = st.text_input("Enter BMI: ")
    DiabetesPegridreeFunction = st.text_input("Enter Diabetes Pedigree Function: ")
    Age = st.text_input("Enter Age: ")
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure,SkinThickness, Insulin, BMI, DiabetesPegridreeFunction, Age])
    
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()