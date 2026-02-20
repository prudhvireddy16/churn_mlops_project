from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Load our saved model
model = XGBClassifier()
model.load_model("models/churn_model.xgb")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    result = "Churn" if prediction[0] == 1 else "Stay"
    return {"prediction": result}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs to see the interactive documentation."}