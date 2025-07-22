# StockMarket-LSTM-Prediction

Este proyecto consiste en la predicción del mercado bursátil utilizando redes neuronales LSTM (Long Short-Term Memory), aplicadas sobre series temporales de datos financieros. Fue desarrollado como parte de un proyecto académico de la materia **Aprendizaje Automático** de la Universidad Tecnológica de Panamá (UTP).

## Descripción

El objetivo del proyecto es construir un modelo que permita predecir precios futuros del mercado de valores a partir de datos históricos. Se utilizan técnicas de Deep Learning, especialmente redes LSTM, por su capacidad para capturar patrones secuenciales y dependencias de largo plazo en series temporales.

El notebook incluye:
- Preparación de datos (normalización, creación de secuencias, divisiones de entrenamiento y prueba).
- Construcción del modelo LSTM.
- Entrenamiento y evaluación.
- Visualización de predicciones.
- Cálculo de métricas de rendimiento.

## Tecnologías y Herramientas

- Python 3.x  
- Jupyter Notebook  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  

## Contenido del Proyecto

- `ML_Project_LSTM.ipynb`: Notebook principal con todo el flujo del proyecto.
- Este `README.md` con la documentación completa.
- Los datos pueden ser cargados desde fuentes externas (CSV, APIs financieras) o simulados dentro del notebook.

## Requerimientos

Instala las siguientes dependencias antes de ejecutar el proyecto:

```
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install tensorflow
```
O en una sola línea:
```
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Instrucciones de Ejecución

1. Clona este repositorio:
```
git clone https://github.com/tu-usuario/StockMarket-LSTM-Prediction.git
```
2. Ve al directorio del proyecto:
```
cd StockMarket-LSTM-Prediction
```
3. Instala dependencias (si no lo hiciste ya):
```
pip install numpy pandas matplotlib scikit-learn tensorflow
```
4. Abre el notebook:
```
jupyter notebook ML_Project_LSTM.ipynb
```

## Resultados
El notebook genera predicciones sobre precios futuros y las compara con los valores reales. Incluye visualizaciones y métricas como:

- RMSE (Root Mean Square Error)

- R² (Coeficiente de determinación)

- Precisión direccional (aciertos al predecir si el precio sube o baja)

## Mejoras Posibles
- Incluir más variables (volumen, indicadores técnicos, sentimiento de noticias).

- Ajustar hiperparámetros (capas, neuronas, tasa de aprendizaje, epochs).

- Probar arquitecturas híbridas (CNN+LSTM, Transformer para series temporales).

- Validación cruzada temporal.

## Advertencia
Este proyecto es únicamente con fines académicos y educativos. No constituye asesoría financiera ni recomendación de inversión.

#Autor
Julio Lara
Estudiante de Ingeniería en Sistemas y Computación
Universidad Tecnológica de Panamá (UTP)
Materia: Aprendizaje Automático

