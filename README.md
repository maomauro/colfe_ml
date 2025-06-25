# colfe_ml
Proyecto para crear el modelo ML de regresión - predicción de liquidación quincenal de la cooperativa COLFE

## Descripción

Este proyecto implementa un flujo completo de machine learning para predecir el valor de liquidación quincenal de proveedores en la cooperativa COLFE. Utiliza datos históricos de liquidaciones, obtenidos desde un API desarrollado en PHP, para entrenar y actualizar un modelo de regresión con PyCaret.

El proceso incluye:
- Descarga y procesamiento de los datos históricos de liquidación.
- Entrenamiento y reentrenamiento automático del modelo cada vez que se liquida una nueva quincena.
- Exposición del modelo entrenado como un servicio API mediante FastAPI para realizar predicciones sobre la siguiente quincena.
- Integración con sistemas externos (por ejemplo, aplicaciones PHP) para automatizar tanto el reentrenamiento como la consulta de predicciones.

El objetivo es facilitar la toma de decisiones y la planeación financiera, proporcionando estimaciones automáticas y actualizadas del total neto a pagar en cada liquidación

## **Resumen del flujo**

1. **Descargas y filtras los datos históricos**.
2. **Identificas la última quincena liquidada** (para no usarla en el entrenamiento).
3. **Entrenas el modelo solo con datos históricos previos**.
4. **Guardas el mejor modelo** para usarlo luego en predicciones.