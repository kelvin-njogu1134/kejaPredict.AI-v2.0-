#  KejaPredict AI

KejaPredict AI is a machine learning-powered real estate price prediction system focused on the Kenyan property market. It helps users estimate fair rental and sale prices based on location, property type, and housing features such as bedrooms and bathrooms.

---

##  Features

*  Predict property prices using machine learning
*  Location-based pricing (e.g., Kilimani, Westlands, Karen)
*  Supports multiple property features (bedrooms, bathrooms, category)
*  Trained on scraped Kenyan real estate data
*  Interactive Streamlit web app interface
*  Data preprocessing and feature engineering pipeline

---

##  Tech Stack

* Python 
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras (if used in model training)
* Streamlit (UI)
* Requests & BeautifulSoup (data scraping)
* OpenStreetMap Nominatim API (geocoding)

---

##  Project Structure

```
kejaPredict.AI/
│
├── data/                  # Dataset files (CSV)
├── models/                # Trained ML models
├── notebooks/             # Jupyter notebooks (EDA & training)
├── app.py                 # Streamlit app
├── location.py            # Geocoding script
├── preprocess.py          # Data cleaning & preprocessing
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

##  Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/kejaPredict.AI.git
cd kejaPredict.AI
```

2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the App

```bash
streamlit run app.py
```

---

##  How It Works

1. User selects location, bedrooms, bathrooms, and property category
2. System preprocesses inputs
3. ML model predicts estimated price
4. Results are displayed in real-time via Streamlit

---

##  Example Inputs

| Feature   | Example Value |
| --------- | ------------- |
| Location  | Kilimani      |
| Bedrooms  | 3             |
| Bathrooms | 2             |
| Category  | Flats         |

---

##  Model Pipeline

* Data Cleaning (handling missing values)
* Feature Encoding (categorical → numerical)
* Scaling (MinMaxScaler)
* Train/Test Split
* Model Training (Regression model)
* Evaluation (MAE / RMSE)

---

##  Future Improvements

* Add real-time property scraping
* Integrate Google Maps API for better geolocation
* Deploy API backend (FastAPI)
* Improve model accuracy with ensemble methods
* Add user authentication system

---

##  Author

Kelvin Njogu
Machine Learning & AI Developer (Kenya 🇰🇪)

---



 If you like this project, give it a star on GitHub!

