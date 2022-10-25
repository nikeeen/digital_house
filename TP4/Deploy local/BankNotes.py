from pydantic import BaseModel

class BankNote(BaseModel):
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