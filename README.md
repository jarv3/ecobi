# 📊 Econometría con Business Intelligence para la gestión estratégica del crecimiento económico

Aplicación interactiva desarrollada en **Python + Streamlit** para el **análisis macroeconómico aplicado**, integrando datos oficiales del **Banco Mundial** con técnicas de **econometría de series temporales**, **modelos multivariantes** y **Machine Learning**.

Este proyecto está diseñado para **profesionales de economía, finanzas**, así como para **uso académico** en cursos de econometría aplicada, macroeconomía y forecasting.

---

## 🎯 Objetivo del proyecto

Brindar una herramienta práctica y didáctica que permita:

- Analizar indicadores macroeconómicos internacionales
- Evaluar propiedades estadísticas de las series
- Realizar proyecciones con modelos ARIMA
- Estudiar relaciones dinámicas entre variables (correlación y VAR)
- Comparar enfoques econométricos clásicos con Machine Learning
- Facilitar la enseñanza y el análisis profesional basado en datos

---

## 📚 Documentación

📘 **Manual Profesional y Docente (PDF)**  
Incluye explicación metodológica, interpretación económica y guía académica.

👉 `docs/Manual_Profesional_Docente_Dashboard_Econometrico_BI.pdf`

> **Recomendado leer el manual antes de utilizar la aplicación**, especialmente en contextos académicos.

---

## 🧠 Componentes analíticos

- **Indicadores** (Banco Mundial – WDI):
  - PIB (US$)
  - Inflación (%)
  - Desempleo (%)
  - Balanza Comercial (% PIB)
  - Crédito interno al sector privado (% PIB)

- **Econometría de series temporales**
  - Prueba ADF (estacionariedad)
  - ACF y PACF
  - ARIMA (1,1,1)
  - Diagnóstico de residuos (Ljung–Box)
  - Validación fuera de muestra

- **Análisis multivariante**
  - Matriz de correlación (Pearson, Spearman, Kendall)
  - Transformaciones: niveles, Δ, Δ%, z-score
  - Modelo VAR (Vector Autorregresivo)

- **Machine Learning**
  - Regresión Lineal
  - Ridge Regression
  - Random Forest
  - Comparación interpretabilidad vs precisión

---

## ▶️ Ejecución de la aplicación

### 1. Requisitos

- Python **3.9 o superior**
- Librerías principales:
  - streamlit
  - pandas
  - numpy
  - statsmodels
  - scikit-learn
  - plotly
  - requests

### 2. Instalación rápida

```bash
pip install -r requirements.txt
