import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Configurar el título de la aplicación y el tema
st.set_page_config(page_title="Predicción de Acciones", page_icon=":chart_with_upwards_trend:")
st.title("Predicción de Acciones")

# Usar columnas para organizar inputs
col1, col2 = st.columns(2)
with col1:
    stock = st.text_input("Ingrese el ID de la Acción", "GOOG")

with col2:
    # Usar un selector de fecha
    fecha_inicio = st.date_input("Fecha de Inicio", datetime.now().replace(year=datetime.now().year-20))

fecha_fin = datetime.now()

# Descargar los datos históricos de la acción
datos_accion = yf.download(stock, fecha_inicio, fecha_fin)

# Cargar el modelo previamente entrenado
modelo = load_model("Stock_prediction_model.keras")

# Mostrar los datos históricos en la aplicación
st.subheader("Datos de la Acción")
st.write(datos_accion)

# Gráfica de los datos históricos descargados
st.subheader("Datos Históricos del Precio de la Acción")
fig_historial = go.Figure()
fig_historial.add_trace(go.Scatter(x=datos_accion.index, y=datos_accion['Close'], mode='lines', name='Precio de Cierre'))
fig_historial.update_layout(height=500, width=700, title_text="Precio Histórico de Cierre de los ultimos 20 años", xaxis_title="Fecha", yaxis_title="Precio de la Acción")
st.plotly_chart(fig_historial)

# Preparar datos para predicciones
longitud_division = int(len(datos_accion)*0.7)
x_test = pd.DataFrame(datos_accion.Close[longitud_division:])
escalador = MinMaxScaler(feature_range=(0,1))
ultimos_100_dias = datos_accion['Close'].iloc[-100:].values.reshape(-1,1)
ultimos_100_dias_escalados = escalador.fit_transform(ultimos_100_dias)
ultimos_100_dias_escalados = ultimos_100_dias_escalados.reshape(1,100,1)
precio_predicho = modelo.predict(ultimos_100_dias_escalados)
precio_predicho = escalador.inverse_transform(precio_predicho)

# Mostrar la predicción en la aplicación
st.subheader("Precio de la Acción Predicho para Mañana")
st.metric(label=f"Precio de cierre predicho para {stock} mañana", value=f"${precio_predicho[0][0]:.2f}")

# Realizar y mostrar predicciones anteriores
datos_escalados = escalador.fit_transform(x_test[['Close']])
x_data = []
y_data = []
for i in range(100, len(datos_escalados)):
    x_data.append(datos_escalados[i-100:i])
    y_data.append(datos_escalados[i])
x_data, y_data = np.array(x_data), np.array(y_data)
predicciones = modelo.predict(x_data)
pred_inv = escalador.inverse_transform(predicciones)
test_inv_y = escalador.inverse_transform(y_data.reshape(-1, 1))

# Crear DataFrame para gráficos
datos_grafico = pd.DataFrame({
    'Datos Originales de Prueba': test_inv_y.flatten(),
    'Predicciones': pred_inv.flatten()
}, index=datos_accion.index[longitud_division+100:])

# Visualización de la comparación de los precios reales y los predichos con Plotly
st.subheader('Precio de Cierre Original vs Precio de Cierre Predicho')
figura = go.Figure()
figura.add_trace(go.Scatter(x=datos_grafico.index, y=datos_grafico['Datos Originales de Prueba'], mode='lines', name='Datos Originales de Prueba'))
figura.add_trace(go.Scatter(x=datos_grafico.index, y=datos_grafico['Predicciones'], mode='lines', name='Predicciones'))
figura.update_layout(height=500, width=700, title_text="Comparación de Precios de la Acción", xaxis_title="Fecha", yaxis_title="Precio de la Acción")
st.plotly_chart(figura)

# Añadir la tabla de comparación de valores originales y predichos
st.subheader("Comparación Detallada")
st.dataframe(datos_grafico)


#streamlit run web_predictor.py