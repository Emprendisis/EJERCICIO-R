
import streamlit as st
import pandas as pd
import numpy as np

# --- LinearRegression import with numpy fallback ---
try:
    from sklearn.linear_model import LinearRegression  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
    import numpy as _np

    class LinearRegression:  # simple 1D fallback
        def __init__(self):
            self.coef_ = _np.array([_np.nan])
            self.intercept_ = _np.nan

        def fit(self, X, y):
            # Expect X shape (n,1), y shape (n,)
            x = _np.asarray(X).reshape(-1)
            y = _np.asarray(y).reshape(-1)
            x_mean = _np.nanmean(x)
            y_mean = _np.nanmean(y)
            # var/cov (with ddof=0 to match population formulas used by sklearn in simple tests)
            denom = _np.nanvar(x)
            if denom == 0 or _np.isnan(denom):
                b = 0.0
            else:
                b = _np.nancov(x, y, ddof=0)[0, 1] / denom
            a = y_mean - b * x_mean
            self.coef_ = _np.array([float(b)])
            self.intercept_ = float(a)
            return self

        def score(self, X, y):
            # R^2 computation
            x = _np.asarray(X).reshape(-1)
            y = _np.asarray(y).reshape(-1)
            y_hat = self.intercept_ + self.coef_[0] * x
            ss_res = _np.nansum((y - y_hat) ** 2)
            ss_tot = _np.nansum((y - _np.nanmean(y)) ** 2)
            if ss_tot == 0:
                return 0.0
            return float(1 - ss_res / ss_tot)

from io import BytesIO
import re

st.set_page_config(page_title="Proyección de Ventas", layout="wide")
st.title("📊 Proyección con selección de variable dependiente")

# ---------- utilidades ----------
def normalize_cols(cols):
    rep = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    out = []
    for c in cols:
        c2 = str(c).translate(rep)
        c2 = re.sub(r"\s+", "", c2)  # sin espacios
        out.append(c2)
    return out

def to_numeric_df(df):
    # quita %,$,comas y convierte a numerico
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def scale_to_percentage_points(df, exclude=("Año","Ano","Year")):
    # si una columna (no excluida) tiene valores en 0..1, multiplica *100
    for c in df.columns:
        if c in exclude:
            continue
        col = df[c]
        if col.dropna().abs().max() <= 1.5:  # dato viene como proporción
            df[c] = col * 100.0
    return df

# ---------- carga de datos ----------
st.subheader("📂 Cargar datos históricos")
st.markdown(
    "Sube **CSV o Excel**. La app acepta **variables en filas o en columnas**.\n"
    "Variables típicas: **PIB, Empleo, TipoCambioPct, Inflacion, Ventas**."
)
up = st.file_uploader("Archivo histórico (CSV/XLSX)", type=["csv","xlsx"])

if not up:
    st.info("Sube primero tu archivo histórico para continuar.")
    st.stop()

# Lee y normaliza
if up.name.endswith(".csv"):
    raw = pd.read_csv(up, header=0)
else:
    raw = pd.read_excel(up, header=0)

raw.columns = normalize_cols(raw.columns)
raw = raw.loc[:, ~raw.columns.duplicated()]  # elimina duplicados exactos

# ¿Vienen años como columnas y variables en filas?
need_transpose = False
norm_first_col = normalize_cols(list(raw.iloc[:,0])) if len(raw.columns) > 0 else []
if "Ventas" not in raw.columns and "Ventas" in norm_first_col:
    raw = raw.set_index(raw.columns[0])
    need_transpose = True

if need_transpose:
    df = raw.transpose().reset_index().rename(columns={"index":"Año"})
else:
    df = raw.copy()

df.columns = normalize_cols(df.columns)
# Si no hay columna Año, intenta inferirla del índice
if "Año" not in df.columns and "Ano" not in df.columns and "Year" not in df.columns:
    df.insert(0, "Año", range(1, len(df)+1))

# convierte a numérico y escala a puntos porcentuales
df = to_numeric_df(df)
df = scale_to_percentage_points(df)

st.caption("Datos históricos (tras limpieza/normalización). Porcentajes en **puntos porcentuales (0–100)**.")
st.dataframe(df, use_container_width=True)

# ---------- Paso 1: Identificar variables y preguntar por la dependiente ----------
st.subheader("🧭 Selecciona la variable dependiente a pronosticar")
all_cols = [c for c in df.columns if c not in ["Año","Ano","Year"]]
default_y = "Ventas" if "Ventas" in all_cols else (all_cols[-1] if all_cols else None)
y_col = st.selectbox("Variable dependiente (y)", options=all_cols, index=(all_cols.index(default_y) if default_y in all_cols else 0) if all_cols else 0)

# Botón de confirmación para "detenerse y preguntar"
if "confirm_y" not in st.session_state:
    st.session_state.confirm_y = False

col_a, col_b = st.columns([1,3])
with col_a:
    if st.button("Continuar", help="Confirma la variable dependiente y continúa con el análisis"):
        st.session_state.confirm_y = True

with col_b:
    st.info("La app se detendrá hasta que confirmes la variable dependiente.")

if not st.session_state.confirm_y:
    st.stop()

# ---------- Validación de columnas requeridas (se respeta la lógica original) ----------
# Nota: mantenemos los predictores originales porque el flujo posterior depende de ellos y del sidebar.
req = ["PIB","Empleo","TipoCambioPct","Inflacion", y_col]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {missing}. Renombra en tu archivo y vuelve a subir.")
    st.stop()

# ---------- regresiones simples ----------
X_vars = ["PIB","Empleo","TipoCambioPct","Inflacion"]
y = df[y_col].astype(float).values

pend, inter, r2 = {}, {}, {}
corr = {}

for var in X_vars:
    x = df[[var]].astype(float).values
    model = LinearRegression().fit(x, y)
    pend[var] = float(model.coef_[0])
    inter[var] = float(model.intercept_)
    r2[var]   = float(model.score(x, y))
    corr[var] = float(pd.Series(df[var]).corr(pd.Series(df[y_col])))

res = pd.DataFrame({
    "Variable": X_vars,
    "Correlacion": [corr[v] for v in X_vars],
    "Pendiente (β)": [pend[v] for v in X_vars],
    "Interseccion (α)": [inter[v] for v in X_vars],
    "R²": [r2[v] for v in X_vars]
})
st.subheader(f"📈 Correlaciones y regresiones simples (y = {y_col})")
st.dataframe(res, use_container_width=True)

# ---------- sidebar: pronósticos ----------
st.sidebar.header("🔮 Escenarios (pronósticos macro)")
st.sidebar.caption("Ingresa valores en **puntos porcentuales** (ej. 2.5 = 2.5%).")

pib_f   = st.sidebar.number_input("Variación PIB (%)",      value=2.50)
des_f   = st.sidebar.number_input("Empleo (%)",             value=3.90)
tc_f    = st.sidebar.number_input("Tipo de cambio (%)",     value=0.28)
infl_f  = st.sidebar.number_input("Inflación (%)",          value=4.80)

forecast = {"PIB":pib_f, "Empleo":des_f, "TipoCambioPct":tc_f, "Inflacion":infl_f}

# ---------- pronósticos para y_col ----------
y_simple = {v: inter[v] + pend[v]*forecast[v] for v in X_vars}

total_r2 = sum(r2.values()) if sum(r2.values()) != 0 else 1.0
pesos = {v: r2[v]/total_r2 for v in X_vars}
y_ponderada = sum(y_simple[v]*pesos[v] for v in X_vars)

pred_tbl = pd.DataFrame({
    "Variable": X_vars,
    f"Pronostico {y_col} (%)": [y_simple[v] for v in X_vars],
    "Peso (R²)": [pesos[v] for v in X_vars]
})

st.subheader(f"📊 Pronóstico de {y_col} con Regresiones Simples")
st.dataframe(pred_tbl.style.format({f"Pronostico {y_col} (%)":"{:.3f}","Peso (R²)":"{:.3f}"}), use_container_width=True)

st.subheader("⚖️ Pronóstico (regresión múltiple ponderada por R²)")
st.metric(f"{y_col} proyectado (%)", f"{y_ponderada:.3f}")

# ---------- descarga ----------
out = BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as w:
    df.to_excel(w, sheet_name="Datos_Historicos", index=False)
    res.to_excel(w, sheet_name="Regresiones", index=False)
    pred_tbl.to_excel(w, sheet_name="Pronosticos", index=False)

st.download_button(
    "📥 Descargar resultados (Excel)",
    data=out.getvalue(),
    file_name="Resultados_Proyecciones.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
