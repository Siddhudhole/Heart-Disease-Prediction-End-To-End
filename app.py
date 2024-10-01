import os 
import sys 
import pickle 
import pandas as pd 
import numpy as np  
import streamlit as st 
from src.mlProject.pipelines.prediction_pipeline import Prediction


st.markdown("<h1 style='text-align: center; color: red;'>Heart Disease Detector</h1>", unsafe_allow_html=True)
st.markdown('----------------------------------------------------------------') 

#age,gender,impluse,pressurehight,pressurelow,glucose,kcm,troponin,
col1, col2, col3 ,col4 = st.columns(4) 

with col1:
   age = st.number_input('Enter Age',min_value=0,max_value=120)
   gender = st.selectbox('Select Gender',('Male','Female')) 

with col2:
   pressurehight = st.number_input('Enter Pressure High')
   pressurelow = st.number_input('Enter Pressure Low') 
   

with col3:
   impluse = st.number_input('Enter Impluse')
   glucose = st.number_input('Enter Glucose') 

with col4: 
   kcm = st.number_input('Enter Kcm')
   troponin = st.number_input('Enter Troponin')   
if gender == 'Female':
   gender = 0
elif gender == 'Male':
   gender = 1 


if 'key' not in st.session_state:
    st.session_state.key = Prediction() 

if st.button(label='Predict'):
   with st.spinner('Predicting...'):
    # simulate the long-running task by using a spinner
   
    if age != 0 and pressurehight !=0 and pressurelow !=0 and  impluse !=0 and glucose != 0 :
            data = pd.DataFrame(np.asarray([age,gender,impluse,pressurehight,pressurelow,glucose,kcm,troponin]).reshape(1,-1),
                                columns=['age','gender','impluse','pressurehight','pressurelow','glucose','kcm','troponin'])  
            result = st.session_state.key.predict(data)  
        
            if result[0]=='negative':
                st.success('You do not have heart disease')
        
            elif result[0] == 'positive' : 
                st.error('You have heart disease')     
        
    

    else :
        st.error('Please fill all the fields') 