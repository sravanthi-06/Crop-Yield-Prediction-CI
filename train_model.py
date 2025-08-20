# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("crop_production.csv")
df = df.dropna(subset=['Area','Production'])
df = df[df['Area'] > 0]
df['Yield'] = df['Production'] / df['Area']

# Use a smaller sample for faster CI runs
df_small = df.sample(5000, random_state=42)

# Features & target
num_cols = ['Crop_Year']
cat_cols = ['State_Name', 'Season', 'Crop']

X = df_small[num_cols + cat_cols]
y = df_small['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
}

best_model, best_name, best_r2 = None, None, -999

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{name}: R²={r2:.4f} | RMSE={rmse:.4f}")

    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, pipe

print(f"\n✅ Best Model: {best_name} with R²={best_r2:.4f}")

# Save best model
joblib.dump(best_model, "best_yield_model.pkl")
