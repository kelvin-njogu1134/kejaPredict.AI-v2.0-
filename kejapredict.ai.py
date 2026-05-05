import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="KejaPredict AI", layout="wide")

# -----------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("keja_xgb_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

model, columns = load_model()

# -----------------------
# SESSION STATE
# -----------------------
if 'step' not in st.session_state:
    st.session_state.step = 1

def gopage1(): st.session_state.step = 1
def gopage2(): st.session_state.step = 2
def gopage3(): st.session_state.step = 3
def gopage4(): st.session_state.step = 4

# -----------------------
# PAGE 1 - HOME
# -----------------------
if st.session_state.step == 1:

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.image("logo.jpeg", width=150)

    col1, col2, col3 = st.columns([2, 4, 2])
    with col2:
        st.title("Welcome to KejaPredict.AI")
        st.caption("Your Ultimate Rental Price Prediction Tool")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("About KejaPredict AI", on_click=gopage2)
    with col2:
        st.button("How it Works", on_click=gopage3)
    with col3:
        st.button("Start Predicting", on_click=gopage4)

# -----------------------
# PAGE 2 - ABOUT
# -----------------------
elif st.session_state.step == 2:

    st.title("About KejaPredict AI")

    st.write("""
    KejaPredict AI predicts house prices in Kenya using machine learning 
    and location-based intelligence.
    """)

    st.markdown("""
    ✔ Data-driven pricing  
    ✔ Location-aware predictions  
    ✔ Smart real estate insights  
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back Home", on_click=gopage1)
    with col2:
        st.button("How it Works", on_click=gopage3)

# -----------------------
# PAGE 3 - HOW IT WORKS
# -----------------------
elif st.session_state.step == 3:

    st.title("How KejaPredict AI Works")

    st.markdown("""
    1. Data collection  
    2. Cleaning & preprocessing  
    3. Feature engineering  
    4. Model training  
    5. Prediction  
    """)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=gopage2)
    with col2:
        st.button("Start Predicting", on_click=gopage4)

# -----------------------
# PAGE 4 - PREDICTION
# -----------------------
elif st.session_state.step == 4:

    st.title("Price Prediction")

    # -------- Load dataset (for dropdowns only) --------
    df = pd.read_csv("nairobi_real_estate_data.csv")
    df = df.fillna(0)

    # -------- Clean Locations --------
    df["Location"] = df["Location"].apply(lambda x: x.split(",")[-1].strip())
    areas = sorted(df["Location"].unique().tolist())

    # -------- Extract Categories --------
    category_cols = [col for col in df.columns if col.startswith("Category_")]
    df["Category"] = df[category_cols].idxmax(axis=1)
    df["Category"] = df["Category"].str.replace("Category_", "")
    categories = sorted(df["Category"].unique().tolist())

    # -------- UI --------
    col1, col2 = st.columns(2)

    with col1:
        Location_input = st.selectbox("Select Area", areas)
        no_bedrooms = st.number_input("Bedrooms", 1, 10, 3)

    with col2:
        category_input = st.selectbox("Property Type", categories)
        no_bathrooms = st.number_input("Bathrooms", 1, 10, 2)

    # -------- Location → Coordinates --------
    location_coords = {
        "Kilimani": (-1.2921, 36.8219),
        "Karen": (-1.319, 36.707),
        "Westlands": (-1.267, 36.810),
        "Runda": (-1.224, 36.806),
        "Lavington": (-1.283, 36.783),
    }

    # -------- Prepare Input --------
    def prepare_input(data_dict, columns):
        df = pd.DataFrame([data_dict])
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        return df

    # -------- Predict --------
    if st.button("Predict Price"):

        lat, lon = location_coords.get(Location_input, (-1.2921, 36.8219))

        input_data = {
            "Bedrooms": no_bedrooms,
            "Bathrooms": no_bathrooms,
            "Location": Location_input,
            "Category": category_input,
            "Latitude": lat,
            "Longitude": lon
        }

        new_data = prepare_input(input_data, columns)

        prediction = model.predict(new_data)[0]

        st.success(f"Estimated Price: KES {int(prediction):,}")
        st.warning(f"This may not be real price it is estimation")

    st.divider()

    st.button("⬅ Back Home", on_click=gopage1)