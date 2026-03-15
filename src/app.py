from utils import db_connect
engine = db_connect()

#Cargando los datos

import os
os.system("pip install pandas matplotlib statsmodels pmdarima scikit-learn")
import pandas as pd

url = ("https://breathecode.herokuapp.com/asset/internal-link?id=2546&path=sales.csv")

total_data = pd.read_csv(url)

total_data.head()

# (Ajusta los nombres si el CSV tiene otros)
df = total_data.copy()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Ordenar el índice cronológicamente (muy importante en series temporales)
df = df.sort_index()

print(df.head())

#Construye y analiza la serie temporal

#Para responder a las preguntas, necesitamos visualizar la serie, descomponerla en sus componentes principales y aplicar una prueba estadística (como el test de Dickey-Fuller aumentado) para comprobar la estacionariedad

import matplotlib.pyplot as plt


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# 1. Graficar la serie temporal
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['sales'], label='Ventas')
plt.title('Evolución de las Ventas a lo largo del tiempo')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.show()

# 2. Encontrar el "tensor" (frecuencia de los datos)
frecuencia = df.index.to_series().diff().value_counts().idxmax()
print(f"El tensor de la serie temporal es: {frecuencia}")

# 3. Descomponer la serie (Tendencia, Estacionalidad, Ruido/Residuo)
# Ajusta el 'period' según el tensor (ej. 30 para datos diarios con patrón mensual)
descomposicion = seasonal_decompose(df['sales'], model='additive', period=30)
descomposicion.plot()
plt.show()

# 4. Prueba de Estacionariedad (Test de Dickey-Fuller)
resultado_adf = adfuller(df['sales'])
print(f'P-valor del test ADF: {resultado_adf[1]}')
if resultado_adf[1] < 0.05:
    print("La serie ES estacionaria.")
else:
    print("La serie NO es estacionaria.")

#¿Cuál es el tensor de la serie temporal?
 #El código te lo dirá exactamente, pero revisará la diferencia entre filas. Si hay un dato por día, el tensor es diario (1 día).

#¿Cuál es la tendencia?
 #"ha ido en aumento", #la gráfica mostrará una tendencia alcista (creciente).

#¿Es estacionaria?
 #No. Una serie con una clara tendencia creciente (su media cambia con el tiempo) no es estacionaria. El P-valor del test ADF seguramente será mayor a 0.05, confirmándolo.

#¿Existe variabilidad o presencia de ruido?
 #Sí. Al observar la gráfica de "Resid (Residuos)" en la descomposición, verás fluctuaciones aleatorias que no pueden ser explicadas ni por la tendencia ni por la estacionalidad (ruido).

 #Entrena un ARIMA

 #Para encontrar la mejor parametrización $(p, d, q)$ sin tener que hacer ensayo y error manual observando los gráficos de autocorrelación (ACF/PACF), lo mejor es utilizar la función auto_arima de la librería pmdarima.

import pmdarima as pm

# Dividir en conjunto de entrenamiento y prueba (ej. 80% train, 20% test)
# En series temporales NO se pueden barajar los datos, se cortan cronológicamente
split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]

print(f"Tamaño Train: {len(train)}, Tamaño Test: {len(test)}")

# Entrenar el modelo Auto-ARIMA para buscar los mejores parámetros
# trace=True te mostrará el proceso de búsqueda en la consola
modelo_arima = pm.auto_arima(train['sales'], 
                             start_p=1, start_q=1,
                             test='adf',       # usa adftest para encontrar la 'd' óptima
                             max_p=5, max_q=5, # máximo p y q
                             m=1,              # frecuencia de la serie (1 si no hay estacionalidad clara, o cámbialo a 12 para meses, etc.)
                             d=None,           # deja que el modelo determine 'd'
                             seasonal=False,   # pon True si detectaste estacionalidad anual/mensual clara
                             trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)

print(modelo_arima.summary())

#Predice con el conjunto de test
#Ahora validaremos qué tan bueno es nuestro modelo comparando sus predicciones con el trozo de datos que apartamos (Test).

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Realizar predicciones sobre el periodo de test
predicciones = modelo_arima.predict(n_periods=len(test))
# Asegurarnos de que las predicciones compartan el índice temporal de los datos de prueba
predicciones.index = test.index

# Calcular métricas de rendimiento
mse = mean_squared_error(test['sales'], predicciones)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test['sales'], predicciones)

print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")

# Visualizar Realidad vs Predicción
plt.figure(figsize=(12, 5))
plt.plot(train.index, train['sales'], label='Entrenamiento')
plt.plot(test.index, test['sales'], label='Datos Reales (Test)')
plt.plot(test.index, predicciones, color='red', label='Predicciones ARIMA')
plt.title('Predicción de Ventas vs Datos Reales')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.show()

#Guarda el modelo

import pickle
import os

# Crear la carpeta 'models' si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# Ruta donde se guardará el modelo
ruta_modelo = 'models/modelo_arima_ventas.pkl'

# Guardar el modelo
with open(ruta_modelo, 'wb') as archivo:
    pickle.dump(modelo_arima, archivo)

print(f"Modelo guardado exitosamente en: {ruta_modelo}")

# Nota: Para cargarlo en el futuro, usarías:
# with open('models/modelo_arima_ventas.pkl', 'rb') as archivo:
#     modelo_cargado = pickle.load(archivo)