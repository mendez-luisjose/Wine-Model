import streamlit as st
import pandas as pd
import pickle 
import numpy as np

@st.cache_data
def get_data() :
    dataset = pd.read_csv("./data/winequalityN.csv")

    return dataset

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl('random_forest_model.pkl') 


def prediction(row) :
   X = pd.DataFrame([row], columns=['type', 'fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

   predicted = model.predict(X)

   if predicted == 0 :
    st.success("It is a Regular Wine 💢")
   elif predicted == 1 :
    st.success("It is a Good Wine 💫")
   elif predicted == 2 :
    st.success("It is an Excellent Wine 💥")


header = st.container()
body = st.container()

with header :
    st.title("Wine Model 🍷")
    st.image("./img.jpg")
    st.header("Wine Classification Model with Random Forest")

with body :
    dataset = get_data()

    st.write("The Model predicts the Quality of the Wine, if is Regular One, Good, or Excellent.")
    st.subheader("Dataset Preview: ")
    st.write(dataset.head(15))

    st.subheader("Check It-out!")
    st.write("Please Fill-In all the options for predict the Quality of the Wine: ")

    color = st.selectbox("Color: ", options=["Red", "White"], index=0)
    acidity = st.number_input("Fixed Acidity: ")
    volatile = st.number_input("Volatile Acidity: ")
    citric = st.number_input("Citric acid: ")
    sugar = st.number_input("Residual Sugar: ")
    chlorides = st.number_input("Chlorides: ")
    free_sulfur = st.number_input("Free Sulfur Dioxide: ")
    total_sulfur = st.number_input("Total Sulfur Dioxide: ")
    density = st.number_input("Density: ")
    ph = st.number_input("pH: ")
    sulphates = st.number_input("Sulphates: ")
    alcohol = st.number_input("Alcohol: ")

    if (color == "Red") :
        color = 0
    elif (color == "White") :
        color = 1

    row = np.array([color, acidity, volatile, citric, sugar, chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol])

    st.button("Predict Quality", on_click=prediction, args=(row,))

    st.write("Be aware that the Quality of the Wine is going to appear at top of the page!")

    


