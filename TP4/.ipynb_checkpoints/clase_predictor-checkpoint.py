{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d510d713",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Tratamiento de datos\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "#Visualización de datos\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from xgboost import plot_importance\n",
    "#from matplotlib import rcParams\n",
    "\n",
    "#Análisis, modelado, predicción y métricas\n",
    "from sklearn.model_selection import train_test_split\n",
    "from xgboost.sklearn import XGBRegressor\n",
    "from statsmodels.tsa.arima_model import ARIMA\n",
    "from statsmodels.tsa.stattools import adfuller\n",
    "import statsmodels.api as sm\n",
    "import statsmodels.tsa.api as smt\n",
    "\n",
    "import pickle\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.base import BaseEstimator, TransformerMixin\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "64513ffd",
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyModel():\n",
    "    def __init__(self, model_path:str=\"modelo_entrenado.pkl\"):\n",
    "        with open(model_path, 'rb') as handle:\n",
    "            self.model = pickle.load(handle)\n",
    "            \n",
    "    def predict(self, array):\n",
    "        label = self.model.predict(array)[0]\n",
    "        return(label)\n",
    "                                         "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ae77e30a",
   "metadata": {},
   "outputs": [],
   "source": [
    "mymodel = MyModel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7bb75400",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ec47859",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "94026c9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "prueba = pd.DataFrame({'temp': [19.8], 'timeindex': [4685], 'mes_10': [1], 'mes_11': [0], 'mes_12': [0], \\\n",
    "              'mes_2': [0], 'mes_3': [0], 'mes_4': [0], 'mes_5': [0], 'mes_6': [0], 'mes_7': [0], \\\n",
    "              'mes_8': [0], 'mes_9': [0], 'tipo_dia_HÁBIL': [1], 'tipo_dia_SÁBADO': [0], \\\n",
    "              'estado_tiempo_N': [1], 'estado_tiempo_SN': [0], 'nueva_estacion_otoño': [0], \\\n",
    "              'nueva_estacion_primavera': [1], 'nueva_estacion_verano':[0]})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "48de3e5c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "374.71207"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mymodel.predict(prueba)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26ce5be2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:dhdsblend2021] *",
   "language": "python",
   "name": "conda-env-dhdsblend2021-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
