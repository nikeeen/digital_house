#Tratamiento de datos
import pandas as pd
import numpy as np

#Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
#from matplotlib import rcParams

#Análisis, modelado, predicción y métricas
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt

import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class MyModel():
    def __init__(self, model_path:str="modelo_entrenado.pkl"):
        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle)
            
    def predict(self, array):
        label = self.model.predict(array)[0]
        return(label)
                       
mymodel = MyModel()

