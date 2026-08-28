import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("logistic_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Diabetes Prediction using Logistic Regression")

preg = st.number_input("Pregnancies",0,20)
glucose = st.number_input("Glucose",0,300)
bp = st.number_input("Blood Pressure",0,150)
skin = st.number_input("Skin Thickness",0,100)
insulin = st.number_input("Insulin",0,900)
bmi = st.number_input("BMI",0.0,70.0)
dpf = st.number_input("Diabetes Pedigree Function",0.0,3.0)
age = st.number_input("Age",1,100)

if st.button("Predict"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    data = scaler.transform(data)

    pred = model.predict(data)

    if pred[0] == 1:
        st.error("Patient is likely Diabetic")
    else:
        st.success("Patient is Not Diabetic")
