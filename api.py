import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

mlflow.set_tracking_uri("sqlite:///mlflow.db")

modelo = mlflow.sklearn.load_model("models:/titanic-mlp/latest")

app = FastAPI(title="API Titanic MLP")


class Passageiro(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str


@app.post("/predict")
def predict(passageiro: Passageiro):
    dados = pd.DataFrame([passageiro.model_dump()])
    predicao = modelo.predict(dados)[0]
    return {"prediction": int(predicao)}
