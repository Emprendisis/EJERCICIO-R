
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from statsmodels.api import add_constant, OLS

st.set_page_config(page_title="Regresión para Pronóstico (Ventas)", layout="wide")

st.title("Modelo de Regresión para Pronóstico de Ventas")
st.caption("Carga datos históricos (PIB, Desempleo, Tipo de cambio %, Inflación, **Ventas**). Ingresa pronósticos o carga un archivo con pronósticos.")

# ---------- Utilidades ----------
def read_user_file(uploaded):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        # Excel
        try:
            df = pd.read_excel(uploaded, engine="openpyxl")
        except Exception:
            df = pd.read_excel(uploaded)
    return df

def clean_numeric(df):
    # Forzar numérico donde se pueda
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def autodetect_and_orient(df):
    """
    Reglas:
    - Debe existir la variable objetivo 'Ventas' como columna. 
    - Si no está como columna pero sí como índice -> transponer.
    - Si ninguna contiene 'Ventas', intentar detectar por mayúsculas/minúsculas.
    """
    cols_lower = [c.lower() for c in df.columns]
    if "ventas" in cols_lower:
        # ya está en columnas
        return df.copy()
    # ¿Está en índice?
    if df.index.name and df.index.name.lower() == "ventas":
        return df.T
    # ¿Está en alguna fila como primer columna (variables en filas)?
    # Heurística: si la primera columna parece contener nombres de variables
    first_col = df.columns[0]
    candidates = df[first_col].astype(str).str.lower().tolist()
    if "ventas" in candidates:
        # usar la primera columna como índice y transponer
        tmp = df.set_index(first_col)
        return tmp.T

    # Como último recurso: probar transponer y ver si aparece
    t = df.T
    if "ventas" in [c.lower() for c in t.columns]:
        return t

    return df.copy()

def ensure_columns_order(df):
    # Intenta ordenar dejando 'Ventas' al final para claridad
    cols = list(df.columns)
    ventas_col = None
    for c in cols:
        if c.lower() == "ventas":
            ventas_col = c
            break
    if ventas_col:
        others = [c for c in cols if c != ventas_col]
        return df[others + [ventas_col]]
    return df

def compute_correlations(df, target="Ventas"):
    corr = df.corr(numeric_only=True)
    if target in corr.columns:
        res = corr[[target]].drop(index=target, errors="ignore").rename(columns={target: "Correlación con Ventas"})
        return res
    return pd.DataFrame()

def simple_regression(y, x):
    X = add_constant(x)
    model = OLS(y, X, missing="drop").fit()
    alpha = model.params.get("const", np.nan)
    beta = model.params.drop("const", errors="ignore").iloc[0] if len(model.params) > 1 else np.nan
    r2 = model.rsquared
    return alpha, beta, r2

def regressions_table(df, target="Ventas"):
    indep = [c for c in df.columns if c.lower() != target.lower()]
    rows = []
    for var in indep:
        alpha, beta, r2 = simple_regression(df[target], df[var])
        rows.append({"Variable": var, "α (intersección)": alpha, "β (pendiente)": beta, "R²": r2})
    res = pd.DataFrame(rows).set_index("Variable")
    return res

def forecast_simple(alpha, beta, x_future):
    return alpha + beta * x_future

def weighted_multiple_forecast(simple_forecasts, r2_weights):
    # Ponderación por R² normalizado (si todos R²=0 -> promedio simple)
    w = r2_weights.copy().astype(float)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = pd.Series(np.ones_like(w), index=w.index) / len(w)
    return (simple_forecasts * w).sum()

def build_download(workbook_dict):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet_name, df in workbook_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    return bio.getvalue()

# ---------- Panel de carga ----------
col_data, col_fore = st.columns([2, 1])

with col_data:
    st.subheader("1) Cargar datos históricos")
    data_file = st.file_uploader("CSV o Excel con histórico", type=["csv", "xlsx", "xls"])
    df_raw = read_user_file(data_file)

    if df_raw is not None:
        df = autodetect_and_orient(df_raw)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        df = ensure_columns_order(df)
        df = clean_numeric(df)
        st.write("**Datos interpretados** (se asume que la columna objetivo es `Ventas`):")
        st.dataframe(df, use_container_width=True)
    else:
        df = None
        st.info("Carga un archivo. Debe incluir la columna **Ventas**.")

with col_fore:
    st.subheader("2) Pronósticos")
    mode = st.radio("Selecciona modo de pronóstico", ["Manual (una observación)", "Archivo (varias observaciones)"], index=0)
    forecast_df = None
    manual_inputs = {}

    if df is not None:
        indep_vars = [c for c in df.columns if c.lower() != "ventas"]
        if mode == "Manual (una observación)":
            st.caption("Ingresa los valores pronosticados para cada variable independiente.")
            for var in indep_vars:
                val = st.number_input(f"{var}", value=float(df[var].dropna().iloc[-1]) if df[var].dropna().size else 0.0)
                manual_inputs[var] = val
            if manual_inputs:
                forecast_df = pd.DataFrame([manual_inputs])
        else:
            f_file = st.file_uploader("CSV o Excel con pronósticos de variables independientes", type=["csv", "xlsx", "xls"], key="f_file")
            if f_file is not None:
                tmp = read_user_file(f_file)
                tmp = autodetect_and_orient(tmp)
                # Mantener solo variables independientes conocidas
                if df is not None:
                    tmp = tmp[[c for c in tmp.columns if c in indep_vars]]
                forecast_df = clean_numeric(tmp)

# ---------- Cálculos ----------
if df is not None:
    if "Ventas" not in df.columns:
        st.error("No se encontró la columna 'Ventas' en los datos históricos.")
    else:
        with st.expander("Correlaciones con Ventas", expanded=True):
            corr_tbl = compute_correlations(df, target="Ventas")
            if not corr_tbl.empty:
                st.dataframe(corr_tbl.style.format({"Correlación con Ventas": "{:.4f}"}), use_container_width=True)
            else:
                st.warning("No fue posible calcular correlaciones (verifica que existan datos numéricos).")

        with st.expander("Resultados de regresión simple", expanded=True):
            reg_tbl = regressions_table(df, target="Ventas")
            if not reg_tbl.empty:
                st.dataframe(reg_tbl.style.format({"α (intersección)": "{:.6f}", "β (pendiente)": "{:.6f}", "R²": "{:.6f}"}), use_container_width=True)
            else:
                st.warning("No fue posible estimar las regresiones (revisa los datos).")

        st.subheader("3) Pronósticos")
        if forecast_df is None or forecast_df.empty:
            st.info("Proporciona pronósticos (manual o archivo) para calcular las ventas pronosticadas.")
        else:
            indep_vars = [c for c in df.columns if c.lower() != "ventas"]
            # Pronósticos por variable (regresión simple)
            simple_results = []
            for var in indep_vars:
                alpha = reg_tbl.loc[var, "α (intersección)"]
                beta  = reg_tbl.loc[var, "β (pendiente)"]
                r2    = reg_tbl.loc[var, "R²"]
                col_preds = forecast_df[var].apply(lambda x: forecast_simple(alpha, beta, x))
                simple_results.append(col_preds.rename(var))
            simple_preds = pd.concat(simple_results, axis=1)

            # Pronóstico ponderado por R²
            weights = reg_tbl["R²"]
            weighted = simple_preds.apply(lambda row: weighted_multiple_forecast(row, weights), axis=1)
            out = pd.DataFrame({
                "Pronóstico ponderado por R²": weighted
            })
            # Unir también los pronósticos simples
            out = pd.concat([simple_preds, out], axis=1)

            st.write("**Pronósticos de Ventas**")
            st.dataframe(out, use_container_width=True)

            # Descarga a Excel
            wb = {
                "Datos": df,
                "Correlaciones": corr_tbl if not corr_tbl.empty else pd.DataFrame(),
                "Regresiones": reg_tbl if not reg_tbl.empty else pd.DataFrame(),
                "Pronosticos": out
            }
            xls = build_download(wb)
            st.download_button(
                label="Descargar resultados en Excel",
                data=xls,
                file_name="resultados_regresion_pronostico.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.sidebar.title("Instrucciones")
st.sidebar.markdown("""
1. Carga un **CSV/Excel** con histórico. Debe incluir una columna llamada **Ventas**.  
2. Si tus variables están en filas, la app intenta **transponer** automáticamente.  
3. Ingresa pronósticos **manuales** o carga un **archivo con pronósticos**.  
4. La app calcula **correlaciones**, **β**, **α**, **R²**, **pronósticos simples** y el **pronóstico ponderado por R²**.  
5. Puedes **descargar** todas las tablas en Excel.
""")
