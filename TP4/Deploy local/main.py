from typing import Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import os

#from clase_predictor import MyModel
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt

import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

app = FastAPI()

with open('modelo_entrenado.pkl', 'rb') as f:
    regressor = pickle.load(f)

class items(BaseModel):
    temp: float
    timeindex: float
    mes_10: float
    mes_11: float
    mes_12: float
    mes_2: float
    mes_3: float
    mes_4: float
    mes_5: float
    mes_6: float
    mes_7: float
    mes_8: float
    mes_9: float
    tipo_dia_habil: float
    tipo_dia_sabado: float
    estado_tiempo_N: float
    estado_tiempo_SN: float
    nueva_estacion_otono: float
    nueva_estacion_primavera: float
    nueva_estacion_verano: float

@app.get('/')
async def read_root():
    return {'Trabajo práctico 4 - Grupo 4'}


@app.post('/predictor')

def predict_banknote(data:items):
    data = dict(data)
    df = pd.DataFrame([data.values()], columns=data.keys(), index=[1])
    prediccion = regressor.predict(df)
    
    return {"prediccion": float(prediccion)}

