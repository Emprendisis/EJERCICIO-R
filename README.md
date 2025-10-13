# Proyección con selección de variable dependiente (Streamlit)

App de pronóstico que:
1) Carga un CSV/XLSX y detecta variables.
2) Se detiene para que elijas la **variable dependiente** a pronosticar.
3) Ejecuta correlaciones, regresiones simples contra **PIB, Empleo, TipoCambioPct, Inflacion**, calcula un pronóstico ponderado por R² y exporta a Excel.

## Uso local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura esperada de datos
- Formatos: **CSV** o **XLSX**.
- Porcentajes en **puntos porcentuales** (p. ej., 4.8 = 4.8%).
- Columnas requeridas para el flujo actual: `PIB`, `Empleo`, `TipoCambioPct`, `Inflacion` y la **variable dependiente** que selecciones (p. ej. `Ventas`).

> La app intenta normalizar nombres (quita acentos y espacios).

## Despliegue en Streamlit Cloud
1) Sube `app.py`, `requirements.txt` y **opcional** `runtime.txt` (fija Python 3.11).
2) Crea la app en Streamlit Cloud apuntando a `app.py`.

## Dependencias y fallback
La app usa `scikit-learn` para regresiones simples. Si **no** está disponible, aplica un **fallback con NumPy** (implementación propia de regresión lineal 1D con cálculo de R²). Aun así se recomienda mantener `scikit-learn` en producción.
