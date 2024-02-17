# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 22:48:19 2023

@author: ankit
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model 

# loaded_model = pickle.load(open("model/trained_model.pkl",'rb'))
loaded_model = pickle.load(open('model/trained_model.pkl','rb'))

# creating a function for prediction 

def Heart_disease_Prediction(input_data):
    
    
    # changing data to numpy array 
    input_data_array = np.asarray(input_data, dtype = np.float64)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped =  input_data_array.reshape(1,-1)
    
    result = loaded_model.predict(input_data_reshaped)
    print("The prediction is : ",result)
    
    if (result[0] == 1):
      return "The person has Heart Diseases"        
    else:
      return "The person has not Heart Diseases"
  

def main():
    # giving a title 
    st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Prediction Application</h1>", unsafe_allow_html=True)
    
    # getting the input data from input user
    
    age= st.text_input("Age of person : ")
    sex= st.text_input("Sex : ")
    
    cp = st.text_input("Chest pain type : ")
    restbps = st.text_input("Resting BP : ")
    chol = st.text_input("Serum Cholestoral (mg/dl) : ")
    fbs = st.text_input("Fasting blood sugar > 120 mg/dl : ")
    restecg = st.text_input("Resting electrocardiographic results (0-2) : ")
    
    thalach = st.text_input("Maximum heart rate achieved : ")
    exang = st.text_input("Exercise induced angina : ")
    oldpeak = st.text_input("Oldpeak : ")
    slope = st.text_input("Slope of the peak exercise ST segment : ")
    ca = st.text_input("Number of major vessels (0-3) : ")
    thal = st.text_input("chest pain type : ")
    
    
    # code for prediction 
    predict = '' # null string 
    
    # creating a button for prediction 
    
    if st.button('Diagnosis Test Result'):
        predict = Heart_disease_Prediction([age, sex, cp, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal ])
        
    st.success(predict)
    
    st.markdown("***")
    
    st.text("\n\n")
#    st.markdown("<h3 style='text-align: center; color: red;'> Model accuracy is   </h3>", unsafe_allow_html=True)    
    
    st.write(" \n\n\n\n")
    st.markdown("******")
    
    st.write("Contributor : [Vishank Goel & Sai Vaibhav](https://github.com/HitThe5ky)")
    
    st.write("\n© 2023 Heart Disease Prediction System. All rights reserved.")
if __name__ == '__main__':
    main()
    
    
