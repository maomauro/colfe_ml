import pandas as pd
from pycaret.regression import setup, compare_models, save_model

# Descargar datos históricos
url = "http://localhost:82/colfe_web/api/apiTotalLiquidacion.php"
df = pd.read_json(url)
df_proveedor = df[df['vinculacion'] == 'proveedor'].copy()

# Identificar la última quincena liquidada
ultima = df_proveedor['fecha_liquidacion'].max()
df_historico = df_proveedor[df_proveedor['fecha_liquidacion'] < ultima]
df_modelo = df_historico[['quincena', 'total_litros', 'total_neto']]

# Entrenar y guardar el modelo
reg1 = setup(
    data=df_modelo,
    target='total_neto',
    session_id=123,
    normalize=True,
    silent=True
)
top5 = compare_models(n_select=5)
save_model(top5[0], 'modelo_liquidacion')
print("Modelo reentrenado y guardado.")