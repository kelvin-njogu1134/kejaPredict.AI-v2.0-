import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# PAGE CONFIG

st.set_page_config(page_title="KejaPredict AI", layout="wide")


# SESSION STATE

if 'step' not in st.session_state:
    st.session_state.step = 1

def gopage1():
    st.session_state.step = 1

def gopage2():
    st.session_state.step = 2

def gopage3():
    st.session_state.step = 3

def gopage4():
    st.session_state.step = 4


# PAGE 1 - HOME

if st.session_state.step == 1:

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
     st.image("logo.jpeg", width=150)

    col1,col2,col3 = st.columns([2,4,2])

    with col2:
     st.title("Welcome to KejaPredict.AI ")
     st.caption("Your Ultimate Rental Price Prediction Tool")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("About KejaPredict AI", on_click=gopage2, key="home_about")

    with col2:
        st.button("How it Works", on_click=gopage3, key="home_how")

    with col3:
        st.button("Start Predicting", on_click=gopage4, key="home_predict")




    


# PAGE 2 - ABOUT

elif st.session_state.step == 2:

    st.title("About KejaPredict AI")

    st.write("""
    KejaPredict AI is a machine learning-based real estate intelligence system that predicts house prices in Kenya using data-driven insights.

    It combines scraped property data with geospatial features (latitude and longitude) to understand how location and property characteristics influence pricing.
    """)

    st.subheader(" Problem It Solves")
    st.markdown("""
    - Unclear and inconsistent property pricing in Kenya  
    - Lack of data-driven decision making for buyers and renters  
    - Scattered and unstructured property listings  
    - Location impact often ignored  
    """)

    st.subheader(" Solution")
    st.markdown("""
    ✔ Scraping real estate data  
    ✔ Cleaning and structuring dataset  
    ✔ Using OpenStreetMap API for coordinates  
    ✔ Machine learning price prediction  
    """)

    st.subheader(" Impact")
    st.markdown("""
    - Helps buyers understand fair prices  
    - Reduces overpaying  
    - Improves investment decisions  
    - Better real estate insights in Kenya  
    """)

    st.success("Built with Python | Streamlit | Machine Learning | OpenStreetMap API")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.button("⬅ Back Home", on_click=gopage1, key="about_back")

    with col2:
        st.button("How it Works", on_click=gopage3, key="about_next")

    st.divider()
    st.markdown("### Built by Njogu 👨‍💻 | Machine Learning Engineer")

    st.markdown("""
    ### 🌐 Connect with me
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
    <a href="https://github.com/kelvin-njogu1134/KejaPredict.AI">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="30"/>
    </a>
    GitHub
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <a href="https://linkedin.com/">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30"/>
    </a>
    LinkedIn
    """, unsafe_allow_html=True)

    with col3:
     st.markdown("""
    <a href="https://instagram.com/">
        <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30"/>
    </a>
    Instagram
    """, unsafe_allow_html=True)


# PAGE 3 - HOW IT WORKS

elif st.session_state.step == 3:

    st.title(" How KejaPredict AI Works")

    st.write("""
    KejaPredict AI uses scraping, cleaning, geospatial analysis, and machine learning to predict real estate prices.
    """)

    st.subheader("1. Data Collection")
    st.markdown("""
    - Location  
    - Price  
    - Bedrooms  
    - Bathrooms  
    - Property type  
    """)

    st.subheader("2.Data Cleaning")
    st.markdown("""
    - Remove duplicates  
    - Handle missing values  
    - Standardize formats  
    """)

    st.subheader("3. Geospatial Processing")
    st.markdown("""
    - Convert locations → Latitude & Longitude  
    - Using OpenStreetMap API  
    """)

    st.subheader("4.  Feature Engineering")
    st.markdown("""
    - Bedrooms  
    - Bathrooms  
    - Location coordinates  
    - Property type encoding  
    """)

    st.subheader("5.  Model Training")
    st.markdown("""
    - Learn patterns between features and price  
    """)

    st.subheader("6.  Prediction")
    st.markdown("""
    - Predict house prices from user inputs  
    """)

    st.success("KejaPredict AI turns raw data into smart predictions ")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.button("⬅ Back", on_click=gopage2, key="how_back")

    with col2:
        st.button("Start Predicting", on_click=gopage4, key="how_next")

    st.divider()

    st.markdown("### Built by Njogu 👨‍💻 | Machine Learning Engineer")

    st.markdown("""
    ### 🌐 Connect with me
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
    <a href="https://github.com/kelvin-njogu1134/KejaPredict.AI">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="30"/>
    </a>
    GitHub
    """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <a href="https://linkedin.com/">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30"/>
    </a>
    LinkedIn
    """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
    <a href="https://instagram.com/">
        <img src="https://cdn-icons-png.flaticon.com/512/2111/2111463.png" width="30"/>
    </a>
    Instagram
    """, unsafe_allow_html=True)


# PAGE 4 - PREDICTION (PLACEHOLDER)

elif st.session_state.step == 4:

    #PREDICTION
        



    st.title(" Price Prediction")

    st.write("This is where your ML model will be connected.")

    df = pd.read_csv('nairobi_real_estate_data.csv')
    df.fillna(0)

    areas = df['Location'].unique().tolist()
    areas = df["Location"].apply(lambda x: x.split(",")[-1].strip())
    areas = sorted(areas.unique().tolist())

    category_cols = [col for col in df.columns if col.startswith("Category_")]
    df["Category"] = df[category_cols].idxmax(axis=1)
    df["Category"] = df["Category"].str.replace("Category_", "")
    categories = df["Category"].unique().tolist()
    


    Location_input = st.selectbox("Select area",areas)
    category_input = st.selectbox("Select Property Type", categories)

    no_bedrooms = st.number_input("Enter number of bedrooms",min_value=0, max_value=10)

    no_bathrooms = st.number_input("Enter number of bathrooms", min_value=0 , max_value=10)


    
    if st.button("Predict now"):
        if not all ([Location_input,category_input,no_bedrooms,no_bathrooms]):
            st.warning("Please fill inputs")

        else:
           st.success("Success")






        
        
        
        
        st.divider()

    st.button("⬅ Back Home", on_click=gopage1, key="predict_back")