from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

def data_preparation(df: pd.DataFrame) -> pd.DataFrame:
    df_prepared = df.copy()
    df_prepared.columns = df_prepared.columns.str.replace("_", " ").str.title()
    df_prepared.rename(columns={"Bmi": "BMI"}, inplace=True)
    df_prepared = df_prepared.dropna()
    df_prepared.columns = df_prepared.columns.str.replace("_", " ").str.title()
    df_prepared.rename(columns={"Bmi": "BMI"}, inplace=True)
    df_prepared = df_prepared.apply(
        lambda col: col.astype("category") if col.dtype == "object" else col
    )
    binary_columns = ["Hypertension", "Heart Disease", "Smoking Status"]
    df_prepared[binary_columns] = df_prepared[binary_columns].apply(
        lambda col: col.astype("category")
    )
    columns_with_categories_to_rename = ["Hypertension", "Heart Disease"]
    for column in columns_with_categories_to_rename:
        df_prepared[column] = df_prepared[column].cat.rename_categories(
            {0: "No", 1: "Yes"}
        )
    df_prepared = df_prepared[(df_prepared["BMI"] <= 65)]
    df_prepared = df_prepared[df_prepared["Gender"] != "Other"]
    df_prepared = df_prepared[df_prepared["Work Type"] != "Never_worked"]
    df_prepared = df_prepared[df_prepared["Age"] > 35]
    df_prepared["Smoking Status"] = df_prepared["Smoking Status"].replace(
        {"formerly smoked": "smoked", "smokes": "smoked"}
    )
    return df_prepared

with open("trained_model.pkl", "rb") as f:
    model = joblib.load(f)

app = FastAPI()

class InputData(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict/")
def predict(data: InputData):
    input_data = pd.DataFrame([data.dict()])
    prepared_data = data_preparation(input_data)
    prediction = model.predict(prepared_data)
    return {"prediction": int(prediction[0])}
