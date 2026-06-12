import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load Model and Scaler
model = load_model("iris_ann_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="ANN Iris Classifier",
    page_icon="🌸"
)

st.title("🌸 Iris Flower Classification using ANN")

st.write(
    "Enter flower measurements and predict the species."
)

# Inputs
sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    value=0.2
)

if st.button("Predict"):

    data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    class_index = np.argmax(prediction)

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower: {classes[class_index]}"
    )

    st.write("Prediction Probabilities")

    st.write({
        classes[i]: float(prediction[0][i])
        for i in range(3)
    })