from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import os
from typing import Dict

app = FastAPI()

# Global variables to store the model, scaler, and dataset
model = None
scaler = None
dataset = None

# Pydantic model for prediction input
class PredictInput(BaseModel):
    Temperature: float
    Runtime: float

# Endpoint 1: Upload CSV dataset
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global dataset
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        # Read CSV file into a Pandas DataFrame
        df = pd.read_csv(file.file)
        required_columns = {"Machine_ID", "Temperature", "Runtime", "Failures"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV file must contain columns: {required_columns}",
            )
        dataset = df
        return {"message": "Dataset uploaded successfully.", "rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

# Endpoint 2: Train the model
@app.post("/train")
async def train_model():
    global model, scaler, dataset
    if dataset is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Upload a dataset first.")

    try:
        # Prepare the data
        X = dataset[["Temperature", "Runtime"]]
        y = dataset["Failures"]
        
        # Handle imbalanced data with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Standardize the features
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )

        # Train the model
        model = SVC(probability=True)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Save the model and scaler
        joblib.dump(model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        return {
            "message": "Model trained successfully.",
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "precision":precision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# Endpoint 3: Predict a single input
@app.post("/predict")
async def predict(input_data: PredictInput):
    global model, scaler
    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Train the model first.")

    try:
        # Preprocess input
        input_df = pd.DataFrame([input_data.dict()])
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0].max()

        return {
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": round(confidence, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run this app using the command below:
# uvicorn script_name:app --reload
