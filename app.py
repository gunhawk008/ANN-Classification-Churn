import tensorflow as tf
import numpy as np
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the training model

model = tf.keras.models.load_model('model.h5')

#load Encoder , scaler 
with open('label_encoder_gender', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encode_geography', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler', 'rb') as file:
    scaler = pickle.load(file)

# StreamLit App

st.title("Customer Churn Prediction")

#User Input

geography = st.selectbox('Geography',onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',18,65)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input("estimated_salary")
tenure = st.slider("Tenure",0,20)
num_of_products = st.slider('Num of product',1,4)
has_credit_card = st.selectbox('has_credit_card',[0,1])
is_active = st.selectbox('is active',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]


st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
