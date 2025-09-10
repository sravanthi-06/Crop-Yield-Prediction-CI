# test_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

df = pd.read_csv("crop_production.csv")

def test_not_empty():
    assert len(df) > 0, "Dataset is empty!"

def test_no_nulls():
    assert df[['Area','Production']].isnull().sum().sum() == 0, "Data contains nulls!"

def test_model_performance():
    df_clean = df[df['Area'] > 0].copy()
    df_clean['Yield'] = df_clean['Production'] / df_clean['Area']

    num_cols = ['Crop_Year']
    cat_cols = ['State_Name', 'Season', 'Crop']

    X = df_clean[num_cols + cat_cols]
    y = df_clean['Yield']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    model = joblib.load("best_yield_model.pkl")
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    assert r2 > 0.3, f"R² too low: {r2:.2f}"
