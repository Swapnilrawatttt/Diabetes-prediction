# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open("D:/Model deployement/diabetes_model.sav", 'rb'))


# Collect user input
a = int(input("Enter No. Pregnancies: "))
b = int(input("Enter Glucose Level: "))
c = int(input("Enter Blood Pressure level: "))
d = int(input("Enter Skin Thickness: "))
e = int(input("Enter Insulin Level: "))
f = float(input("Enter BMI: "))
g = float(input("Enter Diabetes Pedigree Function: "))
h = int(input("Enter Age: "))

# Change input data to numpy array
input_data = np.asarray([a, b, c, d, e, f, g, h])

# Reshape the array for predicting one instance
input_data_reshaped = input_data.reshape(1, -1)

# Standardize the input data
#13std_data = scaler.transform(input_data_reshaped)

# Make prediction
prediction = loaded_model.predict(input_data_reshaped)

# Output result
if prediction[0] == 0:
    print("No Diabetes")
else:
    print("The person is Diabetic")