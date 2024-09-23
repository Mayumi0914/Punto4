import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import predict_model
import tempfile

# Cargar el modelo preentrenado
with open('best_model.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

dataset = pd.read_csv( "dataset_APP.csv") 

# Título de la API
st.title("API de Predicción price")


# Botón para subir archivo Excel
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

if st.button("Predecir"):
    if uploaded_file is not None:
        try:
            # Cargar el archivo subido
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            if uploaded_file.name.endswith(".csv"):
                prueba = pd.read_csv(tmp_path)
            else:
                prueba = pd.read_excel(tmp_path)

            cuantitativas = ['Avg. Session Length','Time on App','Time on Website','Length of Membership']
            categoricas = ['dominio', 'Tec']

            base_modelo = pd.concat([prueba.get(cuantitativas),prueba.get(categoricas)],axis = 1)

            # Realizar predicción
            df_test = base_modelo.copy()
            predictions = predict_model(dt2, data=df_test)
            predictions["price"] = predictions["prediction_label"]

        
                   # Preparar archivo para descargar
            kaggle = pd.DataFrame({'email': prueba["Email"], 'price': predictions["price"]})

            # Mostrar predicciones en pantalla
            st.write("Predicciones generadas correctamente!")
            st.write(kaggle)

            # Botón para descargar el archivo de predicciones
            st.download_button(label="Descargar archivo de predicciones",
                               data=kaggle.to_csv(index=False),
                               file_name="kaggle_predictions.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Por favor, cargue un archivo válido.")

# Botón para reiniciar la página
if st.button("Reiniciar"):
    st.experimental_rerun()

