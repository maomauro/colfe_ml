from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.regression import load_model, predict_model

app = FastAPI()
modelo = load_model('modelo_liquidacion')

class DatosQuincena(BaseModel):
    quincena: str
    total_litros: float

@app.post("/predecir/")
def predecir_liquidacion(data: DatosQuincena):
    df = pd.DataFrame([data.dict()])
    resultado = predict_model(modelo, data=df)
    prediccion = resultado['prediction_label'].iloc[0]
    return {"prediccion_total_neto": prediccion}