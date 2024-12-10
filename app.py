import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data_frame = pd.read_csv('./healthcare-dataset-stroke-data.csv')
data_frame['Rural/Urban'] = data_frame['Residence_type'].map({'Rural': 0, 'Urban': 1})
data_frame['Never_Worked'] = 0
data_frame['Private'] = 0
data_frame['gender'] = data_frame['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})

# Replace NaN values in the bmi column with a linear sequence
num_nan = data_frame['bmi'].isna().sum()
bmi_min = data_frame['bmi'].min()
bmi_max = data_frame['bmi'].max()
linear_sequence = np.linspace(bmi_min, bmi_max, num=num_nan)
data_frame.loc[data_frame['bmi'].isna(), 'bmi'] = linear_sequence

# Prepare the features and target variable
X = data_frame[['gender', 'age', 'hypertension', 'heart_disease', 'Rural/Urban', 'avg_glucose_level', 'bmi']]
y = data_frame['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Stroke Prediction App')
st.write('Enter the details of the patient to predict the likelihood of stroke.')

# User inputs
gender = st.selectbox('Gender', options=[0, 1, 2], format_func=lambda x: ['Male', 'Female', 'Other'][x])
age = st.number_input('Age', min_value=0, max_value=120, value=30)
hypertension = st.selectbox('Hypertension', options=[0, 1], format_func=lambda x: ['No', 'Yes'][x])
heart_disease = st.selectbox('Heart Disease', options=[0, 1], format_func=lambda x: ['No', 'Yes'][x])
rural_urban = st.selectbox('Residence Type', options=[0, 1], format_func=lambda x: ['Rural', 'Urban'][x])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=100.0)
height = st.number_input('Height (in cm)', min_value=50.0, max_value=250.0, value=170.0)
weight = st.number_input('Weight (in kg)', min_value=10.0, max_value=300.0, value=70.0)

# Calculate BMI
height_m = height / 100  # Convert height to meters
bmi = weight / (height_m ** 2)

# Prediction
if st.button('Predict'):
    input_data = np.array([[gender, age, hypertension, heart_disease, rural_urban, avg_glucose_level, bmi]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('The model predicts that the patient is likely to have a stroke.')
    else:
        st.write('The model predicts that the patient is not likely to have a stroke.')