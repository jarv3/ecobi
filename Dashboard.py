# --------------------------------------------------------------------------------------------------------------
# Modelo econométrico integrado con Business Intelligence para la gestión estratégica del crecimiento económico 
# --------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, acf, pacf, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
st.set_page_config(
    page_title="Indicadores Macroeconómicos – Banco Mundial",
    layout="wide",
    page_icon="📊"
)
st.title("📊 Sistema Econométrico y Business Intelligence para el Análisis Predictivo y la Gestión Estratégica del Crecimiento Económico")
st.markdown("Fuente: Banco Mundial  \
Proyección: ARIMA  \
Gráficos interactivos Plotly")
st.markdown("---")

# ------------------------------
# Indicadores
# ------------------------------
INDICATORS = {
    "PIB (US$)": "NY.GDP.MKTP.CD",          # USD actuales
    "Inflación (%)": "FP.CPI.TOTL.ZG",      # variación anual del IPC (%)
    "Desempleo (%)": "SL.UEM.TOTL.ZS",      # tasa de desempleo (%)
    "Balanza Comercial (% PIB)": "NE.RSB.GNFS.ZS",  # % del PIB
    "Crédito interno al sector privado (% PIB)": "FS.AST.PRVT.GD.ZS"  # nuevo indicador financiero
    }

# ------------------------------
# Países por continentes
# ------------------------------
CONTINENTS = {
    "América del Sur": {
        "Ecuador": "ECU", "Colombia": "COL", "Perú": "PER", "Chile": "CHL",
        "Argentina": "ARG", "Brasil": "BRA", "Uruguay": "URY", "Paraguay": "PRY",
        "Bolivia": "BOL", "Venezuela": "VEN",
    },
    "América del Norte": {"Estados Unidos": "USA", "Canadá": "CAN", "México": "MEX"},
    "Europa": {"España": "ESP", "Alemania": "DEU", "Francia": "FRA", "Italia": "ITA", "Reino Unido": "GBR"},
    "Asia": {"China": "CHN", "Japón": "JPN", "India": "IND", "Corea del Sur": "KOR", "Singapur": "SGP"},
    "África": {"Sudáfrica": "ZAF", "Nigeria": "NGA", "Egipto": "EGY", "Kenya": "KEN", "Marruecos": "MAR"},
    "Oceanía": {"Australia": "AUS", "Nueva Zelanda": "NZL"},
}

# ------------------------------
# Creación de los filtros
# ------------------------------
st.sidebar.header("🎛️ Filtros")
continent_selected = st.sidebar.selectbox("Selecciona el continente", list(CONTINENTS.keys()))
country_name = st.sidebar.selectbox("Selecciona el país", list(CONTINENTS[continent_selected].keys()))
country_code = CONTINENTS[continent_selected][country_name]
indicator_name = st.sidebar.selectbox("Selecciona el indicador", list(INDICATORS.keys()))
indicator_code = INDICATORS[indicator_name]
start_year = st.sidebar.number_input("Año inicial", min_value=1960, max_value=2024, value=2000, step=1)
end_year = st.sidebar.number_input("Año final", min_value=1960, max_value=2024, value=2024, step=1)
st.sidebar.caption("Consejo: usa un rango ≥ 10 años para una proyección ARIMA razonable.")
enable_multi = st.sidebar.checkbox("Comparar varios países en la misma gráfica")
selected_countries = []
if enable_multi:
    selected_countries = st.sidebar.multiselect(
        "Países adicionales (mismo continente)",
        options=[p for p in CONTINENTS[continent_selected].keys() if p != country_name],
        default=[]
    )

# ---- Opciones para el análisis de series ----
st.sidebar.header("⚙️ Opciones de análisis de series")
nlags = st.sidebar.number_input("Lags para ACF/PACF y Ljung–Box", min_value=5, max_value=60, value=24, step=1)
n_test = st.sidebar.number_input("Años para validación fuera de muestra", min_value=1, max_value=20, value=5, step=1)
adf_on_diff = st.sidebar.checkbox("ADF en primera diferencia", value=True, help="Recomendado para series con tendencia.")
alpha_ci = 0.05  # nivel para bandas de confianza de ACF/PACF

# ------------------------------
# API del banco mundial
# ------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wb_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Descarga datos desde la API del Banco Mundial con paginación."""
    base_url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 1000}
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame(columns=["year", "value"])  # seguridad
    meta, rows = data[0], data[1]
    total_pages = int(meta.get('pages', 1))
    all_rows = list(rows) if isinstance(rows, list) else []
    for page in range(2, total_pages + 1):
        params_page = dict(params); params_page["page"] = page
        rp = requests.get(base_url, params=params_page, timeout=30)
        rp.raise_for_status()
        dp = rp.json()
        if len(dp) >= 2 and isinstance(dp[1], list):
            all_rows.extend(dp[1])
    df = pd.DataFrame([{"date": it.get("date"), "value": it.get("value")} for it in all_rows])
    df["year"] = pd.to_numeric(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.dropna(subset=["year", "value"]).sort_values("year").reset_index(drop=True)
    return df

# ------------------------------
# Obtención de los datos
# ------------------------------
try:
    df_main = fetch_wb_indicator(country_code, indicator_code, int(start_year), int(end_year))
except Exception as e:
    st.error(f"Error al consultar la API del Banco Mundial: {e}")
    df_main = pd.DataFrame(columns=["year", "value"])  # evitar NameError

if df_main.empty:
    st.warning("No se encontraron datos para el filtro seleccionado.")
    st.stop()

# Métricas (siempre actualizado)
st.subheader(f"📌 Métricas clave: {indicator_name} – {country_name} ({start_year}–{end_year})")

# Limpieza de datos para evitar errores
df_main = df_main.drop_duplicates(subset=["year"]).dropna(subset=["year", "value"])
df_main["year"] = pd.to_numeric(df_main["year"], errors="coerce").astype(int)
if not df_main.empty:
    current_year = int(df_main["year"].max())
    current_row = df_main.loc[df_main["year"] == current_year, "value"]
    current_value = float(current_row.iloc[0]) if not current_row.empty else np.nan
    mean_value = float(np.nanmean(df_main["value"].values))
    prev_year = current_year - 1
    variation_pct = np.nan
    previous_row = df_main.loc[df_main["year"] == prev_year, "value"]
    if not previous_row.empty:
        previous_value = float(previous_row.iloc[0])
        variation_pct = ((current_value - previous_value) / previous_value) * 100 if previous_value != 0 else np.nan
    projected_value = np.nan
    variation_vs_projected = np.nan
    if len(df_main) >= 10:
        ts_tmp = df_main.set_index("year")["value"].astype(float)
        try:
            model_tmp = ARIMA(ts_tmp, order=(1, 1, 1))
            res_tmp = model_tmp.fit()
            fc_tmp = res_tmp.forecast(steps=1)
            projected_value = float(fc_tmp.iloc[0])
            variation_vs_projected = ((projected_value - current_value) / current_value) * 100 if current_value != 0 else np.nan
        except Exception as e:
            st.warning(f"No se pudo calcular la proyección ARIMA de 1 año: {e}")
    if indicator_name == "PIB (US$)":
        current_value /= 1_000_000
        mean_value   /= 1_000_000
        projected_value = projected_value / 1_000_000 if not np.isnan(projected_value) else np.nan
        unidad = "millones US$"
    else:
        unidad = ""  # para indicadores en %
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Valor actual", f"{current_value:,.2f}")
    c2.metric("Media histórica", f"{mean_value:,.2f}")
    c3.metric("Variación anual (%)", f"{variation_pct:,.2f}" if not np.isnan(variation_pct) else "N/D")
    c4.metric("Proyectado (ARIMA)", f"{projected_value:,.2f}" if not np.isnan(projected_value) else "N/D")
    c5.metric("Variación vs Proyectado (%)", f"{variation_vs_projected:,.2f}" if not np.isnan(variation_vs_projected) else "N/D")
else:
    st.warning("No hay datos válidos para calcular las métricas.")

# ------------------------------
# Gráficos interáctivos
# ------------------------------
st.markdown("---")
def make_main_figure(df_plot: pd.DataFrame, title: str, y_label: str, series_name: str):
    """Figura principal (un solo país), sin rangeslider y eje X categórico."""
    df_plot = df_plot.copy()
    df_plot["year_str"] = df_plot["year"].astype(str)
    fig = px.line(
        df_plot, x="year_str", y="value",
        title=title,
        labels={"year_str": "Año", "value": y_label},
    )
    fig.update_traces(line=dict(width=3), mode="lines+markers", name=series_name)
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        legend_title=None,
        xaxis=dict(type="category", showgrid=True, 
            gridcolor="#3A4A5A", 
            zerolinecolor="#3A4A5A"),
        yaxis=dict(showgrid=True, gridcolor="#3A4A5A", 
            zerolinecolor="#3A4A5A"),
        xaxis_rangeslider=dict(visible=False)
    )
    return fig
def make_compare_figure(frames: list, title: str, y_label: str):
    """Figura comparativa multi-país, sin rangeslider y eje X categórico."""
    df_comp = pd.concat(frames, ignore_index=True)
    df_comp["year_str"] = df_comp["year"].astype(str)
    fig = px.line(
        df_comp,
        x="year_str", y="value", color="country",
        title=title,
        labels={"year_str": "Año", "value": y_label, "country": "País"},
    )
    fig.update_traces(line=dict(width=3), mode="lines+markers")
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        legend_title="País",
        xaxis=dict(type="category", showgrid=True, 
            gridcolor="#3A4A5A", 
            zerolinecolor="#3A4A5A"),
        yaxis=dict(showgrid=True, 
            gridcolor="#3A4A5A", 
            zerolinecolor="#3A4A5A"),
        xaxis_rangeslider=dict(visible=False)
    )
    return fig
# Lógica de visualización
if enable_multi:
    if selected_countries:
        frames = [df_main.assign(country=country_name)]
        for cname in selected_countries:
            ccode = CONTINENTS[continent_selected][cname]
            try:
                df_cmp = fetch_wb_indicator(ccode, indicator_code, int(start_year), int(end_year))
                if not df_cmp.empty:
                    frames.append(df_cmp.assign(country=cname))
                else:
                    st.warning(f"No hay datos disponibles para {cname} en el rango {start_year}–{end_year}.")
            except Exception as e:
                st.warning(f"No se pudo cargar {indicator_name} para {cname}: {e}")
        if len(frames) > 1:
            fig_comp = make_compare_figure(
                frames,
                title=f"{indicator_name} – Comparación ({start_year}–{end_year})",
                y_label=indicator_name
            )
            st.plotly_chart(fig_comp, width="stretch", config={"displayModeBar": True, "responsive": True})
        else:
            st.info("No hay suficientes países con datos para mostrar la comparación.")
    else:
        st.info("Selecciona al menos un país adicional para mostrar la gráfica comparativa.")
else:
    fig_main = make_main_figure(
        df_main,
        title=f"{indicator_name} – {country_name} ({start_year}–{end_year})",
        y_label=indicator_name,
        series_name=country_name
    )
    st.plotly_chart(fig_main, width="stretch", config={"displayModeBar": True, "responsive": True})

# ------------------------------
# Proyección ARIMA
# ------------------------------
st.markdown("---")
st.subheader("🔮 Proyección ARIMA (próximos 5 años)")
if len(df_main) >= 10:
    ts = df_main.set_index('year')['value'].astype(float)
    ts.index = pd.to_datetime(ts.index, format='%Y')
    try:
        model = ARIMA(ts, order=(1, 1, 1))
        res = model.fit()
        steps = 5
        fc = res.forecast(steps=steps)
        # fc_years = list(range(int(ts.index.max()) + 1, int(ts.index.max()) + 1 + steps))
        fc_years = [(ts.index[-1] + pd.DateOffset(years=i)).year for i in range(1, steps+1)]
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=ts.index.astype(str), y=ts.values,
            mode="lines+markers", name="Histórico",
            line=dict(color="#0EA5E9", width=3), marker=dict(size=6)
        ))
        fig_fc.add_trace(go.Scatter(
            x=[str(y) for y in fc_years], y=fc.values,
            mode="lines+markers", name="Proyección",
            line=dict(color="#EF4444", width=3), marker=dict(size=7)
        ))
        fig_fc.update_layout(
            title=f"ARIMA (1,1,1) – {indicator_name} – {country_name}",
            xaxis_title="Año", yaxis_title=indicator_name,
            template="plotly_white", hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A"),
            yaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A")
            )
        st.plotly_chart(fig_fc, width="stretch", config={"displayModeBar": True, "responsive": True})
        st.caption(f"AIC del modelo: **{res.aic:.2f}**")
    except Exception as e:
        st.warning(f"No fue posible ajustar ARIMA: {e}")
else:
    st.info("Se requieren al menos ~10 observaciones no nulas para una proyección ARIMA razonable.")

# -------------------------------------------------------------
# Análisis del modelo: ADF, ACF/PACF, ARIMA (1,1,1), Residuos, Validación
# -------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Análisis de la serie temporal")

# Serie base (anual) como float, índice de año
ts_all = df_main.set_index("year")["value"].astype(float).sort_index().dropna()

# ---------- 1) Prueba de estacionalidad (ADF) ----------
st.markdown("### 1) Prueba de estacionalidad (ADF)")
serie_adf = ts_all.diff().dropna() if adf_on_diff else ts_all
try:
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, adf_icbest = adfuller(serie_adf.values, autolag="AIC")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estadístico ADF", f"{adf_stat:,.3f}")
    c2.metric("p-valor", f"{adf_p:,.4f}")
    c3.metric("Lags usados", f"{adf_lags}")
    c4.metric("Observaciones", f"{adf_nobs}")
    st.caption(f"Valores críticos: 1%={adf_crit['1%']:.3f} | 5%={adf_crit['5%']:.3f} | 10%={adf_crit['10%']:.3f}")
    st.info("Criterio: si el **p-valor < 0.05**, se rechaza la hipótesis nula de raíz unitaria (serie estacionaria).")
except Exception as e:
    st.warning(f"No fue posible ejecutar ADF: {e}")

# ---------- 2) Gráficos ACF y PACF ----------
st.markdown("### 2) ACF y PACF")
serie_corr = ts_all.diff().dropna() if adf_on_diff else ts_all
N = len(serie_corr)
max_pacf_lags = max(1, min(nlags, N // 2))   # PACF: ≤ 50% de la muestra
max_acf_lags  = max(1, min(nlags, N - 1))    # ACF: capamos por seguridad
if N < 5:
    st.info("Muy pocos datos para estimar ACF/PACF. Amplía el rango de años o reduce lags.")
else:
    try:
        # Cálculo
        acf_vals, acf_conf = acf(serie_corr.values, nlags=max_acf_lags, alpha=alpha_ci, fft=True)
        pacf_vals, pacf_conf = pacf(serie_corr.values, nlags=max_pacf_lags, alpha=alpha_ci, method="ywadjusted")
        # Figura ACF
        x_lags = list(range(len(acf_vals)))
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=x_lags, y=acf_vals, name="ACF", marker_color="#0EA5E9"))
        fig_acf.add_hline(y=0, line=dict(color="#666", width=1))
        fig_acf.update_layout(
            title=f"ACF – {'Δ' if adf_on_diff else ''}{indicator_name} ({country_name})",
            xaxis_title="Lag", yaxis_title="Autocorrelación",
            template="plotly_white", hovermode="x",
            yaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A")
            )
        # Figura PACF
        x_lags_p = list(range(len(pacf_vals)))
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=x_lags_p, y=pacf_vals, name="PACF", marker_color="#EF4444"))
        fig_pacf.add_hline(y=0, line=dict(color="#666", width=1))
        fig_pacf.update_layout(
            title=f"PACF – {'Δ' if adf_on_diff else ''}{indicator_name} ({country_name})",
            xaxis_title="Lag", yaxis_title="Autocorrelación parcial",
            template="plotly_white", hovermode="x",
            yaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A")
        )
        c_acf, c_pacf = st.columns(2)
        with c_acf:
            st.plotly_chart(fig_acf, width="stretch", config={"displayModeBar": True, "responsive": True})
            st.caption("Bandas de confianza calculadas al 95%.")
        with c_pacf:
            st.plotly_chart(fig_pacf, width="stretch", config={"displayModeBar": True, "responsive": True})
            st.caption("Bandas de confianza calculadas al 95%.")
    except Exception as e:
        st.info("No fue posible calcular ACF/PACF con los parámetros actuales. Ajusta los lags o revisa los datos.")
        
# ---------- 3) Ajuste ARIMA (1,1,1) + AIC/BIC ----------
st.markdown("### 3) Ajuste ARIMA (1,1,1) y criterios AIC/BIC")
if len(ts_all) >= 10:
    try:
        model_full = ARIMA(ts_all, order=(1, 1, 1))
        res_full = model_full.fit()
        c1, c2 = st.columns(2)
        c1.metric("AIC", f"{res_full.aic:,.2f}")
        c2.metric("BIC", f"{res_full.bic:,.2f}")
        st.caption("Menor AIC/BIC indica mejor trade-off ajuste vs. complejidad. Compara con otros órdenes si lo deseas.")
    except Exception as e:
        st.warning(f"No se pudo ajustar ARIMA(1,1,1) sobre toda la serie: {e}")
else:
    st.info("Se requieren al menos ~10 observaciones para un ajuste ARIMA razonable.")

# ---------- 4) Diagnóstico de residuos y prueba Ljung–Box ----------
st.markdown("### 4) Diagnóstico de residuos y prueba Ljung–Box")
try:
    # Verifica que el modelo ARIMA sobre toda la serie se haya ajustado
    if 'res_full' not in locals():
        st.info("Primero debe ajustarse el modelo ARIMA para analizar residuos.")
    else:
        # Residuos del modelo
        resid = pd.Series(res_full.resid).dropna()
        N = len(resid)
        # Regla práctica para evitar errores:
        # - Ljung–Box requiere suficiente muestra
        # - Capamos lags a min(nlags elegido, N//2, 40)
        max_lags = int(max(1, min(nlags, N // 2, 40)))
        if max_lags < 1 or N < 5:
            st.info("La serie de residuos es muy corta para Ljung–Box. Reduce lags o amplía el rango de años.")
        else:
            lag_list = list(range(1, max_lags + 1))
            # IMPORTANTE: usar el índice del DataFrame devuelto (no la columna 'lag')
            lb = acorr_ljungbox(resid, lags=lag_list, return_df=True)
            lags_x = lb.index.values              # ← evita KeyError: 'lag'
            # Gráfico de p-valores por lag
            fig_lb = go.Figure()
            fig_lb.add_trace(go.Bar(x=lags_x, y=lb["lb_pvalue"], name="p-valor", marker_color="#22C55E"))
            fig_lb.add_hline(y=0.05, line=dict(color="#EF4444", width=2, dash="dash"),
                             annotation_text="α=0.05", annotation_position="top right")
            fig_lb.update_layout(title="Ljung–Box sobre residuos (H0: no autocorrelación)",
                                 xaxis_title="Lag", yaxis_title="p-valor", template="plotly_white")
            st.plotly_chart(fig_lb, width="stretch",
                            config={"displayModeBar": True, "responsive": True})
            # Interpretación
            if (lb["lb_pvalue"] > 0.05).all():
                st.success("No se rechaza H0 en todos los lags: los residuos parecen no autocorrelados (adecuado).")
            else:
                st.warning("Se rechaza H0 en algunos lags: podrían persistir autocorrelaciones (considera ajustar el orden).")
except Exception as e:
    st.warning(f"No fue posible ejecutar el diagnóstico de residuos: {e}")

# ---------- 5) Validación fuera de muestra (forecast) ----------
st.markdown("### 5) Validación fuera de muestra (últimos años como prueba)")
if len(ts_all) > n_test + 5:
    try:
        years_sorted = ts_all.index.astype(int).tolist()
        # Primer año de prueba es el (max_year - n_test + 1)
        cutoff_year = int(ts_all.index.max()) - int(n_test) + 1
        train = ts_all.loc[ts_all.index.astype(int) < cutoff_year]
        test  = ts_all.loc[ts_all.index.astype(int) >= cutoff_year]
        model_oos = ARIMA(train, order=(1, 1, 1))
        res_oos = model_oos.fit()
        fc_oos = res_oos.forecast(steps=len(test))
        mae = float(np.mean(np.abs(test.values - fc_oos.values)))
        rmse = float(np.sqrt(np.mean((test.values - fc_oos.values)**2)))
        mape = float(np.mean(np.abs((test.values - fc_oos.values) / np.where(test.values==0, np.nan, test.values))) * 100)
        c_mae, c_rmse, c_mape = st.columns(3)
        c_mae.metric("MAE", f"{mae:,.2f}")
        c_rmse.metric("RMSE", f"{rmse:,.2f}")
        c_mape.metric("MAPE (%)", f"{mape:,.2f}")
        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(
            x=train.index.astype(str), y=train.values,
            mode="lines+markers", name="Entrenamiento", line=dict(color="#0EA5E9", width=3)))
        fig_oos.add_trace(go.Scatter(
            x=test.index.astype(str), y=test.values,
            mode="lines+markers", name="Real (Prueba)", line=dict(color="#F59E0B", width=3)))
        fig_oos.add_trace(go.Scatter(
            x=test.index.astype(str), y=fc_oos.values,
            mode="lines+markers", name="Forecast ARIMA", line=dict(color="#EF4444", width=3)))
        fig_oos.update_layout(
            title=f"Validación fuera de muestra – {indicator_name} – {country_name}",
            xaxis_title="Año", yaxis_title=indicator_name,
            template="plotly_white", hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A"),
            yaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A")
        )
        st.plotly_chart(fig_oos, width="stretch", config={"displayModeBar": True, "responsive": True})
        st.caption(f"Entrenamiento hasta {int(train.index.max())}, prueba {len(test)} años desde {int(test.index.min())}.")
    except Exception as e:
        st.warning(f"No fue posible realizar la validación fuera de muestra: {e}")
else:
    st.info("Aumenta el periodo o reduce 'Años para validación' para ejecutar la validación fuera de muestra.")

# ------------------------------
# HEATMAP de correlación
# ------------------------------
st.markdown("---")
st.subheader("📊 Correlación entre indicadores (heatmap interactivo)")
default_corr_inds = ["PIB (US$)", "Balanza Comercial (% PIB)", "Inflación (%)", "Desempleo (%)","Crédito interno al sector privado (% PIB)"]
corr_selection = st.multiselect(
    "Elige los indicadores a correlacionar",
    options=list(INDICATORS.keys()),
    default=default_corr_inds,
    help="Selecciona 2 o más indicadores para generar la matriz de correlación."
)
corr_method = st.selectbox(
    "Método de correlación",
    options=["pearson", "spearman", "kendall"],
    index=0,
    help="Pearson (lineal), Spearman/Kendall (rangos, más robustas)."
)
transform_choice = st.selectbox(
    "Transformación previa de las series",
    options=["Sin transformación", "Crecimiento % vs. año previo", "Estandarización (z-score)"],
    index=0,
    help="El crecimiento % revela co‑movimientos; z-score estandariza niveles."
)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_wb_multi(country_code: str, indicator_map: dict, start_year: int, end_year: int) -> pd.DataFrame:
    """Descarga múltiples indicadores y los consolida por 'year'."""
    frames = []
    for name, code in indicator_map.items():
        df_i = fetch_wb_indicator(country_code, code, start_year, end_year)
        if not df_i.empty:
            df_i = df_i.rename(columns={"value": name})[["year", name]]
            frames.append(df_i)
    if not frames:
        return pd.DataFrame(columns=["year"])
    df_wide = frames[0]
    for k in range(1, len(frames)):
        df_wide = df_wide.merge(frames[k], on="year", how="inner")  # años comunes
    return df_wide.sort_values("year").reset_index(drop=True)
if len(corr_selection) < 2:
    st.info("Selecciona al menos **2** indicadores para generar el heatmap de correlación.")
else:
    selected_map = {name: INDICATORS[name] for name in corr_selection}
    df_wide = fetch_wb_multi(country_code, selected_map, int(start_year), int(end_year))
    if df_wide.empty or df_wide.shape[1] < 3:
        st.warning("No hay suficientes datos para calcular la correlación en el rango seleccionado.")
    else:
        df_corr = df_wide.copy()
        cols = [c for c in df_corr.columns if c != "year"]
        if transform_choice == "Crecimiento % vs. año previo":
            for c in cols:
                df_corr[c] = df_corr[c].pct_change() * 100.0
        elif transform_choice == "Estandarización (z-score)":
            for c in cols:
                x = df_corr[c].astype(float)
                mu, sigma = np.nanmean(x), np.nanstd(x)
                df_corr[c] = (x - mu) / (sigma if sigma not in [0, np.nan] else 1.0)
        df_corr = df_corr.dropna(subset=cols)
        if df_corr.shape[0] < 3:
            st.warning("Se requieren al menos **3 observaciones** tras la transformación.")
        else:
            corr_matrix = df_corr[cols].corr(method=corr_method)
            fig_corr = px.imshow(
                corr_matrix.values,
                x=cols, y=cols,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                text_auto=".2f",
                aspect="auto",
                title=f"Matriz de correlación – {country_name} ({start_year}–{end_year}) [{corr_method}]"
            )
            fig_corr.update_layout(
                template="plotly_white", hovermode="closest",
                coloraxis_colorbar=dict(title="ρ", ticks="outside"),
                xaxis_title="Indicadores", yaxis_title="Indicadores"
            )
            fig_corr.update_coloraxes(cmid=0)
            st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": True, "responsive": True})

# ------------------------------------------------------------
# MODELO VAR (Vector Autorregresivo)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Modelo VAR (Vector Autorregresivo) – sistema de cuatro indicadores")

# Indicadores base (los cuatro solicitados)
var_indicators = ["PIB (US$)", "Balanza Comercial (% PIB)", "Inflación (%)", "Desempleo (%)", "Crédito interno al sector privado (% PIB)"]
var_map = {name: INDICATORS[name] for name in var_indicators}

# Controles en UI
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    var_transform = st.selectbox(
        "Transformación para estacionariedad",
        options=["Crecimiento % (Δ%)", "Primera diferencia (Δ)", "Log-diferencia (Δlog solo PIB)"],
        index=0,
        help="VAR asume series (aprox.) estacionarias. Δ% suele funcionar bien para datos anuales."
    )
with col_v2:
    var_max_lags = st.number_input(
        "Máximo número de rezagos para seleccionar (AIC/BIC)",
        min_value=1, max_value=5, value=2, step=1,
        help="Para datos anuales suele bastar 1–2 rezagos."
    )
with col_v3:
    var_steps = st.number_input(
        "Horizonte de pronóstico (años)",
        min_value=1, max_value=10, value=5, step=1
    )

# Descarga y preparación (reutiliza tu helper)
df_var_wide = fetch_wb_multi(country_code, var_map, int(start_year), int(end_year))  # ya lo tienes creado
if df_var_wide.empty or df_var_wide.shape[1] < 5:  # 1 columna es 'year' + 4 indicadores
    st.warning("No hay suficientes datos para estimar el VAR con los cuatro indicadores en el rango seleccionado.")
else:
    df_var = df_var_wide.copy()
    cols = [c for c in df_var.columns if c != "year"]
    # Transformaciones para (aprox.) estacionariedad
    if var_transform == "Crecimiento % (Δ%)":
        for c in cols:
            df_var[c] = df_var[c].pct_change() * 100.0
        y_label_suffix = " (Δ%)"
    elif var_transform == "Primera diferencia (Δ)":
        for c in cols:
            df_var[c] = df_var[c].diff()
        y_label_suffix = " (Δ)"
    else:  # Log-diferencia solo al PIB; resto en primera diferencia
        # Log-diff del PIB (≈ crecimiento %)
        df_var["PIB (US$)"] = np.log(df_var["PIB (US$)"]).diff() * 100.0
        for c in cols:
            if c != "PIB (US$)":
                df_var[c] = df_var[c].diff()
        y_label_suffix = " (Δ / Δlog)"

    # Limpieza
    df_var = df_var.dropna(subset=cols).sort_values("year").reset_index(drop=True)

    # Reglas mínimas de muestra para VAR
    if df_var.shape[0] < (var_max_lags + 5):
        st.info("Muy pocas observaciones tras la transformación. Amplía el período o reduce el número máximo de rezagos.")
    else:
        try:
            # Selección automática de rezagos por criterios de información
            model = VAR(df_var[cols])
            sel = model.select_order(maxlags=int(var_max_lags))
            # Usamos AIC como criterio principal; si no está, caemos a BIC/HQIC/FPE
            selected_lag = (
                sel.aic or sel.bic or sel.hqic or sel.fpe or 1
            )
            # Ajuste del VAR con el lag seleccionado
            results = model.fit(selected_lag)
            # Métricas y estabilidad
            c_aic, c_bic, c_stb = st.columns(3)
            with c_aic:
                st.metric("Lag (AIC)", f"{sel.aic if sel.aic is not None else 'N/D'}")
            with c_bic:
                st.metric("Lag (BIC)", f"{sel.bic if sel.bic is not None else 'N/D'}")
            with c_stb:
                st.metric("Estabilidad (roots < 1)", "Sí" if results.is_stable() else "No")
            # Pronóstico multivariado
            fc = results.forecast(y=results.endog[-selected_lag:], steps=int(var_steps))
            fc_df = pd.DataFrame(fc, columns=cols)
            # Años futuros (anuales)
            last_year = int(df_var["year"].max())
            future_years = list(range(last_year + 1, last_year + 1 + int(var_steps)))
            fc_df.insert(0, "year", future_years)
            # Figura: histórico transformado + pronóstico por variable
            fig_var = go.Figure()
            for c in cols:
                # Histórico
                fig_var.add_trace(go.Scatter(
                    x=df_var["year"].astype(str), y=df_var[c],
                    mode="lines+markers", name=f"{c} histórico{y_label_suffix}",
                    line=dict(width=3), marker=dict(size=6)
                ))
                # Pronóstico
                fig_var.add_trace(go.Scatter(
                    x=fc_df["year"].astype(str), y=fc_df[c],
                    mode="lines+markers", name=f"{c} pronóstico{y_label_suffix}",
                    line=dict(width=3, dash="dash"), marker=dict(size=7)
                ))
            fig_var.update_layout(
                title=f"VAR({selected_lag}) – {country_name} – {start_year}–{end_year}",
                xaxis_title="Año",
                yaxis_title=f"Indicadores transformados{y_label_suffix}",
                template="plotly_white",
                hovermode="x unified",
                legend_title="Serie",
                xaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A"),
                yaxis=dict(showgrid=True, gridcolor="#3A4A5A", zeroline=True, zerolinecolor="#3A4A5A")
            )
            st.plotly_chart(fig_var, width="stretch", config={"displayModeBar": True, "responsive": True})
            # Tabla con el pronóstico
            st.caption("Pronóstico VAR en unidades transformadas (según opción elegida).")
            st.dataframe(fc_df, width="stretch")
            # Opcional: Causalidad de Granger (parejas)
            if st.checkbox("Calcular pruebas de causalidad de Granger (parejas)", help="Usa las series transformadas; interpreta con cautela."):
                rows = []
                for x in cols:
                    for y in cols:
                        if x == y:
                            continue
                        try:
                            tests = grangercausalitytests(
                                df_var[[y, x]].dropna(), maxlag=int(selected_lag), verbose=False
                            )
                            pvals = [tests[k][0]['ssr_chi2test'][1] for k in range(1, int(selected_lag)+1)]
                            pv_min = float(np.nanmin(pvals))
                            rows.append({
                                "Hipótesis": f"{x} → {y}",
                                "p-valor mínimo": pv_min,
                                "Conclusión (α=0.05)": "Rechaza H0 (causalidad)" if pv_min < 0.05 else "No rechaza H0"
                            })
                        except Exception as e:
                            rows.append({"Hipótesis": f"{x} → {y}", "p-valor mínimo": np.nan, "Conclusión (α=0.05)": f"Error: {e}"})
                st.dataframe(pd.DataFrame(rows), width="stretch")
        except Exception as e:
            st.warning(f"No fue posible ajustar el VAR: {e}")

# ------------------------------------------------------------
# MACHINE LEARNING (generalizado): Pronóstico 2025 para variable objetivo seleccionada
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🧠 Machine Learning – Pronóstico para variable objetivo (PIB, Inflación, Desempleo, Balanza)")

# 1) Datos base: las 4 series y normalización de 'year'
ml_indicators = ["PIB (US$)", "Balanza Comercial (% PIB)", "Inflación (%)", "Desempleo (%)","Crédito interno al sector privado (% PIB)"]
ml_map = {name: INDICATORS[name] for name in ml_indicators}
df_wide = fetch_wb_multi(country_code, ml_map, int(start_year), int(end_year))
if df_wide.empty or df_wide.shape[1] < 5:  # 'year' + 4 indicadores
    st.warning("No hay suficientes datos para entrenar un modelo ML con las cuatro series.")
else:
    # Garantiza que 'year' sea numérico (soporta NaN)
    df_wide["year"] = pd.to_numeric(df_wide["year"], errors="coerce").astype("Int64")
    # 2) Controles de usuario
    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
    with col_sel1:
        target_var = st.selectbox("Variable objetivo", options=ml_indicators, index=0)
    with col_sel2:
        # Transformaciones del objetivo (dependen del tipo)
        if target_var == "PIB (US$)":
            target_transform = st.selectbox(
                "Transformación del objetivo",
                options=["Crecimiento % (Δ%)", "Primera diferencia (Δ)", "Nivel (US$)"],
                index=0,
                help="Δ% y Δ ayudan a estacionariedad; 'Nivel' modela directamente el PIB."
            )
        else:
            target_transform = st.selectbox(
                "Transformación del objetivo",
                options=["Nivel (%)", "Primera diferencia (Δ)", "Crecimiento % (Δ%)"],
                index=0,
                help="Para variables en %, el 'Nivel' anual suele ser utilizable; si notas tendencia, usa Δ o Δ%."
            )

    with col_sel3:
        feat_transform = st.selectbox(
            "Transformación de predictores",
            options=["Niveles", "Primera diferencia (Δ)", "Crecimiento % (Δ%)"],
            index=0,
            help="Se aplica a predictores (otras variables) y al lag del objetivo."
        )
    with col_sel4:
        lag_k = st.number_input("Rezagos de predictores (k)", min_value=1, max_value=3, value=1, step=1)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_name = st.selectbox("Modelo", options=["Regresión Lineal", "Ridge", "Random Forest"], index=0)
    with col_m2:
        # Hiperparámetros
        if model_name == "Ridge":
            alpha_val = st.number_input("alpha (Ridge)", min_value=0.0, value=1.0, step=0.1)
        elif model_name == "Random Forest":
            n_estimators = st.number_input("n_estimators", min_value=50, max_value=500, value=200, step=50)
            max_depth    = st.number_input("max_depth", min_value=2, max_value=20, value=6, step=1)
    # 3) Construcción de dataset: objetivo y predictores
    df_ml = df_wide.copy().sort_values("year").reset_index(drop=True)
    cols_all = [c for c in df_ml.columns if c != "year"]
    other_vars = [v for v in ml_indicators if v != target_var]
    # --- Objetivo (target) ---
    if target_transform.startswith("Nivel"):
        df_ml["target"] = df_ml[target_var].astype(float)
        target_units = f"Nivel de {target_var}"
    elif "Δ%)" in target_transform:  # Crecimiento %
        df_ml["target"] = df_ml[target_var].pct_change() * 100.0
        target_units = f"Crecimiento % de {target_var}"
    else:  # Primera diferencia
        df_ml["target"] = df_ml[target_var].diff()
        target_units = f"Primera diferencia de {target_var}"
    # --- Predictores (transformados SIN shift aún) ---
    df_feat = pd.DataFrame({"year": df_ml["year"]})
    # Autoregresivo del objetivo (lag del propio objetivo como predictor)
    if target_transform.startswith("Nivel"):
        df_feat["Y_X"] = df_ml[target_var].astype(float)
    elif "Δ%)" in target_transform:
        df_feat["Y_X"] = df_ml[target_var].pct_change() * 100.0
    else:
        df_feat["Y_X"] = df_ml[target_var].diff()
    # Otras variables como predictores
    for c in other_vars:
        if feat_transform == "Niveles":
            df_feat[c+"_X"] = df_ml[c].astype(float)
        elif feat_transform.startswith("Primera diferencia"):
            df_feat[c+"_X"] = df_ml[c].diff()
        else:  # Crecimiento %
            df_feat[c+"_X"] = df_ml[c].pct_change() * 100.0
    # --- Shift para entrenamiento (alinear X(t-k) -> y(t)) ---
    df_feat_shift = df_feat.copy()
    for col in [c for c in df_feat_shift.columns if c != "year"]:
        df_feat_shift[col] = df_feat_shift[col].shift(int(lag_k))
    # --- Conjunto final para modelar (evita 'year' duplicada) ---
    df_model = pd.concat(
        [df_ml[["year", "target"]], df_feat_shift.drop(columns=["year"], errors="ignore")],
        axis=1
    )
    # Normaliza 'year' y limpia nulos
    df_model["year"] = pd.to_numeric(df_model["year"], errors="coerce")
    df_model = df_model.dropna(subset=["year", "target"] + [c for c in df_feat_shift.columns if c != "year"]).copy()
    # Reglas mínimas
    years_numeric = df_model["year"].dropna()
    if years_numeric.empty:
        st.info("No hay años válidos en el dataset tras transformaciones/rezagos.")
        st.stop()
    last_year = int(years_numeric.max())
    if df_model.shape[0] < (n_test + 5):
        st.info("Muy pocas observaciones tras transformaciones y rezagos. Amplía el período o reduce 'Años para validación'.")
    else:
        # 4) Split temporal: entrenamiento vs prueba (últimos n_test años)
        cutoff_year = last_year - int(n_test) + 1
        train_df = df_model.loc[df_model["year"] < cutoff_year].copy()
        test_df  = df_model.loc[df_model["year"] >= cutoff_year].copy()
        X_cols = [c for c in df_feat_shift.columns if c != "year"]
        X_train = train_df[X_cols].values
        y_train = train_df["target"].values
        X_test  = test_df [X_cols].values
        y_test  = test_df ["target"].values
        # 5) Modelo
        if model_name == "Regresión Lineal":
            model = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
        elif model_name == "Ridge":
            model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=float(alpha_val)))])
        else:
            model = RandomForestRegressor(
                n_estimators=int(n_estimators), max_depth=int(max_depth),
                random_state=42, n_jobs=-1
            )
        # 6) Entrenamiento y evaluación
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae  = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mape = float(np.mean(np.abs((y_test - y_pred) / np.where(y_test==0, np.nan, y_test))) * 100)
            r2   = float(r2_score(y_test, y_pred))
            c_mae, c_rmse, c_mape, c_r2 = st.columns(4)
            c_mae.metric("MAE",  f"{mae:,.3f}")
            c_rmse.metric("RMSE", f"{rmse:,.3f}")
            c_mape.metric("MAPE (%)", f"{mape:,.2f}" if np.isfinite(mape) else "N/D")
            c_r2.metric("R²", f"{r2:,.3f}")
            # 7) Gráfica: real vs predicción (en unidades del objetivo transformado)
            fig_ml_ts = go.Figure()
            fig_ml_ts.add_trace(go.Scatter(
                x=test_df["year"].astype(int).astype(str), y=y_test, mode="lines+markers",
                name=f"Real ({target_units})", line=dict(color="#0EA5E9", width=3)
            ))
            fig_ml_ts.add_trace(go.Scatter(
                x=test_df["year"].astype(int).astype(str), y=y_pred, mode="lines+markers",
                name=f"Predicción ({target_units})", line=dict(color="#EF4444", width=3)
            ))
            fig_ml_ts.update_layout(
                title=f"Validación ML – {country_name} ({target_var})",
                xaxis_title="Año", yaxis_title=target_units,
                template="plotly_white", hovermode="x unified"
            )
            st.plotly_chart(fig_ml_ts, width="stretch", config={"displayModeBar": True, "responsive": True})
            # 8) Pronóstico 2025 en nivel (US$ o %), respetando el rezago k
            #    Para 2025 (future_year), usamos predictores del año ref_year = future_year - k.
            future_year = last_year + 1          # 2025 si last_year=2024
            ref_year    = future_year - int(lag_k)
            # df_feat (SIN shift) tiene los predictores crudos por año
            feat_cols = [c for c in df_feat.columns if c != "year"]
            # Asegura que 'year' en df_feat sea numérico
            df_feat["year"] = pd.to_numeric(df_feat["year"], errors="coerce")
            row_future = df_feat.loc[df_feat["year"] == ref_year, feat_cols]
            if row_future.empty:
                st.info("No hay predictores disponibles para el año de referencia. Ajusta 'k' o el rango temporal.")
            else:
                X_future = row_future.values
                y_future_pred = float(model.predict(X_future)[0])
                # Reconstrucción a nivel según transformación del objetivo
                # Tomamos el último nivel observado de la variable objetivo (en last_year)
                last_level_series = df_wide.loc[df_wide["year"] == last_year, target_var]
                if last_level_series.empty:
                    st.info("No se encontró el nivel del último año para reconstrucción. Revisa el rango temporal.")
                else:
                    last_level = float(last_level_series.iloc[0])
                    if target_transform.startswith("Nivel"):
                        level_2025 = y_future_pred
                        units = "US$" if target_var == "PIB (US$)" else "%"
                        st.metric(f"Pronóstico 2025 – {target_var}", f"{level_2025:,.2f} {units}")
                    elif "Δ%)" in target_transform:
                        level_2025 = last_level * (1.0 + y_future_pred/100.0)
                        units = "US$" if target_var == "PIB (US$)" else "%"
                        st.metric(f"Pronóstico 2025 – {target_var}", f"{level_2025:,.2f} {units}", f"{y_future_pred:,.2f}% vs {last_year}")
                    else:  # Primera diferencia (Δ)
                        level_2025 = last_level + y_future_pred
                        units = "US$" if target_var == "PIB (US$)" else "%"
                        delta_sign = "▲" if y_future_pred >= 0 else "▼"
                        st.metric(f"Pronóstico 2025 – {target_var}", f"{level_2025:,.2f} {units}", f"{delta_sign} {abs(y_future_pred):,.2f}")
        except Exception as e:
            st.warning(f"No fue posible entrenar/evaluar el modelo ML: {e}")

# ------------------------------
# DESCARGA DE DATOS (CSV)
# ------------------------------
st.markdown("---")
csv = df_main.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar datos (CSV) – Serie seleccionada",
                   data=csv,
                   file_name=f"{indicator_name}_{country_name}_{start_year}-{end_year}.csv",
                   mime="text/csv")

# ------------------------------
# NOTAS Y AYUDA
# ------------------------------
with st.expander("ℹ️ Notas y consideraciones"):
    st.markdown("""
**Unidades y definiciones:**
- *PIB (US$)*: Valor en millones de dólares actuales (no ajustado por inflación).
- *Inflación (%)*: Variación anual del IPC.
- *Desempleo (%)*: Porcentaje de la fuerza laboral sin empleo.
- *Balanza Comercial (% PIB)*: Saldo de bienes y servicios como % del PIB.
- *Crédito interno al sector privado (% PIB)*: Crédito otorgado al sector privado como % del PIB.

**Interpretación de correlación:**
- **ρ ≈ +1**: Relación positiva fuerte (ambas suben).
- **ρ ≈ -1**: Relación negativa fuerte (una sube, otra baja).
- **Cerca de 0**: Relación débil.
- Prueba **Spearman/Kendall** si sospechas no linealidad u outliers.
- El **crecimiento %** atenúa efectos de nivel y destaca co‑movimientos.

**ARIMA:**
- (1,1,1) es base; compara órdenes alternos con **AIC/BIC**.
- Necesitas ≥ 10 observaciones no nulas para proyecciones razonables.
                
**Modelo VAR:**
- Requiere que las series sean **aprox. estacionarias** → por eso aplicamos transformaciones (Δ%, Δ, Δlog).
- El número de rezagos se selecciona automáticamente con criterios como **AIC/BIC**.
- **Interpretación**:
    - Si el pronóstico está en Δ%: reconstruimos niveles acumulando tasas.
    - Si está en Δ: sumamos cambios absolutos.
- **Ventaja**: captura relaciones dinámicas entre variables (p.ej., cómo inflación afecta PIB).
- **Limitación**: necesita más datos que ARIMA y puede ser sensible a la transformación elegida.

**Modelo Machine Learning:**
- Usa las otras variables como **predictores** y aplica rezagos para evitar fuga de información.
- Modelos disponibles:
    - **Regresión Lineal**: simple y explicativa.
    - **Ridge**: lineal con regularización (reduce sobreajuste).
    - **Random Forest**: captura relaciones no lineales.
- **Validación temporal**: últimos `n_test` años para medir MAE, RMSE, MAPE, R².
- **Interpretación**:
    - Si el objetivo está en Δ%: reconstruimos niveles multiplicando por (1 + Δ%).
    - Si está en Δ: sumamos al último nivel.
    - Si está en nivel: el modelo predice directamente el valor.
- **Ventaja**: flexible, puede incorporar más variables y escenarios hipotéticos.
- **Limitación**: requiere cuidado con rezagos y tamaño de muestra.

**Gráficos interactivos:**
- Usa el **modebar** (cámara para exportar), zoom.

**Fuente:**
- API oficial del **Banco Mundial** (JSON).
""")



