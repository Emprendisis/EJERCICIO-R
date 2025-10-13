
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Selector de Variable Dependiente", layout="centered")
st.title("📂 Cargar datos y seleccionar variable dependiente")

st.markdown("""
**Flujo mínimo:**
1) Sube tu base de datos (**CSV** o **XLSX**).  
2) Se listan las variables detectadas.  
3) Selecciona la **variable dependiente (y)** y la app **se detiene**.
""")

up = st.file_uploader("Archivo de datos (CSV/XLSX)", type=["csv","xlsx"])

if not up:
    st.info("Sube un archivo para continuar.")
    st.stop()

# Leer archivo
try:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)
except Exception as e:
    st.error(f"Error al leer el archivo: {e}")
    st.stop()

st.subheader("Variables detectadas")
cols = list(df.columns)
if not cols:
    st.error("No se detectaron columnas en el archivo.")
    st.stop()

st.write(cols)
st.caption("Vista rápida de los primeros registros:")
st.dataframe(df.head(), use_container_width=True)

# Selección de variable dependiente
st.subheader("Selecciona la variable dependiente (y)")
y_col = st.selectbox("Variable dependiente", options=cols)

# Botón para confirmar y detener
if st.button("Confirmar y detener"):
    st.success(f"Variable dependiente seleccionada: **{y_col}**")
    st.stop()

st.info("Selecciona la variable y pulsa **Confirmar y detener** para finalizar.")
