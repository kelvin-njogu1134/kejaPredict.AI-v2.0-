import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import joblib



# LOAD DATA

df = pd.read_csv("nairobi_real_estate_data.csv")

# Fill missing values
df = df.fillna(0)



# ENCODE LOCATION

df = pd.get_dummies(df, columns=["Location"], drop_first=True)



# FEATURES & TARGET

X = df.drop("Price", axis=1)
y = df["Price"]



# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# MODEL (XGBOOST)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)



# TRAIN

model.fit(X_train, y_train)



# EVALUATION

y_pred = model.predict(X_test)

print("\n--- MODEL EVALUATION ---")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE (KES):", mean_absolute_error(y_test, y_pred))



# SAVE MODEL + COLUMNS

joblib.dump(model, "keja_xgb_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")



# PREDICTION FUNCTION


# Load model (simulate production use)
model = joblib.load("keja_xgb_model.pkl")
columns = joblib.load("model_columns.pkl")

# Example input
new_data = pd.DataFrame([{
    "Bedrooms": 5,
    "Bathrooms": 3,
    "Category_Bedsitter": 0,
    "Category_Commercial": 0,
    "Category_Flats": 1,
    "Category_Houses": 0,
    "Category_Offices": 0,
    "Category_Villas": 0,
    "Latitude": -6.1833368,
    
    "Longitude": 39.2053275,
    "Location_Kilimani": 1
}])

# Ensure all columns match training
new_data = new_data.reindex(columns=columns, fill_value=0)

# Predict
prediction = model.predict(new_data)

print("\n--- PREDICTION ---")
print("Predicted Price (KES):", int(prediction[0]))
