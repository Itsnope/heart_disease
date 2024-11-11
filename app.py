import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model

# Muat model dan scaler
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    lstm_model = load_model('lstm_model.h5')
    with open('svm_classifier.pkl', 'rb') as file:
        svm_classifier = pickle.load(file)
    return scaler, lstm_model, svm_classifier

scaler, lstm_model, svm_classifier = load_models()

# Fungsi untuk melakukan prediksi
def predict_heart_disease(input_data):
    # Scale the input data
    input_data_scaled = scaler.transform([input_data])

    # Reshape untuk input LSTM
    input_data_lstm = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    features = lstm_model.predict(input_data_lstm, verbose=0)

    # Prediksi dengan SVM dan dapatkan probabilitas
    prediction = svm_classifier.predict(features)
    probabilities = svm_classifier.predict_proba(features)
    
    # Membalik prediksi karena label terbalik
    final_prediction = 1 - prediction[0]
    final_probabilities = np.flip(probabilities[0])
    
    return final_prediction, final_probabilities

# Judul aplikasi
st.title("Heart Disease Prediction App")

# Deskripsi aplikasi
st.write("Masukkan data pasien untuk memprediksi kemungkinan penyakit jantung.")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0=False, 1=True)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (0=Normal, 1=ST-T Abnormal, 2=LV Hypertrophy)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("ST Slope (0=Upsloping, 1=Flat, 2=Downsloping)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=fixed defect, 2=normal, 3=reversible defect)", [1, 2, 3])

# Button untuk memprediksi
if st.button("Prediksi", type="primary"):
    # Konversi input ke nilai numerik
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Panggil fungsi prediksi
    result, probs = predict_heart_disease(input_data)
    
    # Tampilkan hasil dengan probabilitas
    st.write("---")
    st.subheader("Hasil Prediksi")

    if result == 1:
        st.error("Prediksi: Pasien kemungkinan memiliki penyakit jantung.")
    else:
        st.success("Prediksi: Pasien kemungkinan tidak memiliki penyakit jantung.")
