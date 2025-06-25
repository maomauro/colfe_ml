from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:82"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = load_model('modelo_liquidacion')

class DatosQuincena(BaseModel):
    quincena: str
    fecha_liquidacion: str
    total_litros: float

@app.post("/predecir/")
def predecir_liquidacion(data: DatosQuincena):
    df = pd.DataFrame([data.dict()])
    resultado = predict_model(modelo, data=df)
    prediccion = resultado['prediction_label'].iloc[0]
    return {
        "quincena": data.quincena,
        "fecha_liquidacion": data.fecha_liquidacion,
        "total_litros": data.total_litros,
        "prediccion_total_neto": prediccion
    }