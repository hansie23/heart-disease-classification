import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
data = pd.read_csv('heart-disease.csv')

# Pick features used by the model
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]

# Load the model
model_file = 'heart_disease_model.pkl'
with open(model_file, 'rb') as file:
    clf = pickle.load(file)

# Streamlit App
st.title('Heart Disease Prediction Web App')

st.markdown('Please enter the following information to predict the likelihood of heart disease:')

# Input fields for features
age = st.number_input('Age', min_value=18, max_value=120, value=55)
sex = st.selectbox('Sex', options=[ 'Male', 'Female'])
cp = st.selectbox('Chest Pain Type', options=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, value=130)
chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=120, max_value=600, value=240)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=['True', 'False'])
restecg = st.selectbox('Resting Electrocardiographic Results', options=['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=70, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', options=['Yes', 'No'])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'])
ca = st.slider('Number of Major Vessels Colored by Fluoroscopy (0-3)', min_value=0, max_value=3, value=0)
thal = st.selectbox('Thal', options=['Normal', 'Fixed defect', 'Reversable defect'])

# Convert inputs to model features
sex = 1 if sex == 'Male' else 0
cp_values = {'Typical angina': 0, 'Atypical angina': 1, 'Non-anginal pain': 2, 'Asymptomatic': 3}
cp = cp_values[cp]
fbs = 1 if fbs == 'True' else 0
restecg_values = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
restecg = restecg_values[restecg]
exang = 1 if exang == 'Yes' else 0
slope_values = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope = slope_values[slope]
thal_values = {'Normal': 2, 'Fixed defect': 1, 'Reversable defect': 0}
thal = thal_values[thal]


# Create input data as a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Make prediction
if st.button('Predict'):
    prediction = clf.predict(input_data)
    
    if prediction[0] == 0:
        st.success('The model predicts: No Heart Disease')
    else:
        st.error('The model predicts: Heart Disease')
    
st.markdown('**Note:** This prediction is based on a machine learning model and should not be considered as a professional medical advice.')
