import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================== Config ==================
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# ================== Load Data & Model ==================
@st.cache_data
def load_data():
    return pd.read_csv('final_datasets.csv')

@st.cache_resource
def load_model():
    with open("RidgeModel.pkl", "rb") as file:
        return pickle.load(file)

data = load_data()
model = load_model()

# ================== Title ==================
st.title("🏠 House Price Prediction App")
st.markdown("This app uses a trained Ridge Regression model to predict **house prices** based on key property features.")

# ================== Sidebar Inputs ==================
st.sidebar.header("📋 Input Property Details")

def user_input_features():
    bedrooms = st.sidebar.selectbox("🛏️ Bedrooms", sorted(data['beds'].unique()))
    bathrooms = st.sidebar.selectbox("🛁 Bathrooms", sorted(data['baths'].unique()))
    size = st.sidebar.selectbox("📏 Size (sq ft)", sorted(data['size'].unique()))
    zip_code = st.sidebar.selectbox("📍 Zip Code", sorted(data['zip_code'].unique()))

    input_dict = {
        'beds': bedrooms,
        'baths': bathrooms,
        'size': size,
        'zip_code': zip_code
    }
    return pd.DataFrame([input_dict])

input_df = user_input_features()

# ================== Input Cleaning ==================
def clean_input(input_df, reference_df):
    input_df = input_df.astype({
        'beds': int,
        'baths': float,
        'size': float,
        'zip_code': int
    })

    # Replace unknowns
    for col in input_df.columns:
        if input_df[col].iloc[0] not in reference_df[col].unique():
            input_df[col] = reference_df[col].mode()[0]

    return input_df

processed_input = clean_input(input_df, data)

# ================== Show Inputs ==================
st.subheader("🔍 Review Your Inputs")
st.dataframe(processed_input)

# ================== Predict Price ==================
if st.button("🔮 Predict Price"):
    prediction = model.predict(processed_input)[0]
    st.success(f"🏷️ Estimated House Price: **${prediction:,.2f}**")
    st.metric(label="Predicted Price", value=f"${prediction:,.0f}")

    # ================== Similar Homes Chart ==================
    st.subheader("📊 Price Distribution of Similar Homes")

    # Check for 'price' column in your dataset
    if 'price' in data.columns:
        similar_homes = data[
            (data['beds'] == processed_input['beds'][0]) &
            (data['baths'] == processed_input['baths'][0]) &
            (data['zip_code'] == processed_input['zip_code'][0])
        ]

        if not similar_homes.empty:
            st.markdown(f"Found {len(similar_homes)} similar listings.")
            st.bar_chart(similar_homes[['size', 'price']].set_index('size'))
        else:
            st.warning("No similar homes found in this zip code for selected features.")
    else:
        st.info("`price` column not found in dataset. Chart disabled.")
# === Custom background color ===
def set_bg_color(color="#f5f7fa"):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background color 
set_bg_color("#e6f2ff")

# ================== Footer ==================
st.markdown("---")
st.markdown(
    "📌 **Note**: This prediction is based on historical data and model training. Actual prices may vary due to unobserved factors."
)
