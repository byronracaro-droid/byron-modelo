import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from fpdf import FPDF
import datetime


def cargar_modelos():
    modelo = joblib.load("modelo.joblib")
    scaler = joblib.load("scaler.joblib")
    metricas = joblib.load("metricas.joblib")
    muestra = joblib.load("muestra.joblib")
    corr = joblib.load("correlaciones.joblib")
    stats = joblib.load("stats.joblib")
    return modelo, scaler, metricas, muestra, corr, stats


def generar_pdf(metricas, mejor):
    fig = px.bar(metricas, x="Modelo", y="R2", color="Modelo")
    pio.write_image(fig, "modelos.png")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Reporte Académico - KDD Viviendas", ln=True)
    pdf.cell(0, 10, f"Fecha: {datetime.datetime.now()}", ln=True)

    pdf.multi_cell(0, 8, f"Mejor modelo: {mejor['Modelo']} (R2={mejor['R2']:.4f})")
    pdf.image("modelos.png", w=150)

    pdf.output("reporte.pdf")


def main():
    st.set_page_config(layout="wide")
    st.title("Plataforma KDD - Predicción de Viviendas")

    modelo, scaler, metricas, muestra, corr, stats = cargar_modelos()

    menu = st.sidebar.radio("Fase KDD", 
        ["Exploración", "Preprocesamiento", "Transformación", "Modelos", "Predicción"]
    )

    # ---------------- EXPLORACIÓN ----------------
    if menu == "Exploración":
        st.header("Exploración de Datos")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(muestra, x="median_house_value", title="Distribución de precios")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(muestra, y="median_house_value", title="Outliers")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Relación entre ingreso y precio"):
            fig = px.scatter(muestra, x="median_income", y="median_house_value", trendline="ols", title="Ingreso medio vs Precio de vivienda")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- PREPROCESAMIENTO ----------------
    elif menu == "Preprocesamiento":
        st.header("Preprocesamiento")
        st.write("Datos divididos en 80% entrenamiento y 20% prueba")
        st.dataframe(stats)

    # ---------------- TRANSFORMACIÓN ----------------
    elif menu == "Transformación":
        st.header("Mapa de correlación")
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- MODELOS ----------------
    elif menu == "Modelos":
        st.header("Comparación de modelos")

        fig = px.bar(metricas, x="Modelo", y="R2", color="Modelo")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metricas)

        mejor = metricas.sort_values("R2", ascending=False).iloc[0]
        st.success(f"Mejor modelo: {mejor['Modelo']} (R²={mejor['R2']:.4f})")

        if st.button("Generar PDF"):
            generar_pdf(metricas, mejor)
            with open("reporte.pdf", "rb") as f:
                st.download_button("Descargar reporte", f, "reporte.pdf")

    # ---------------- PREDICCIÓN ----------------
    elif menu == "Predicción":
        st.header("Predicción con nuevos datos")

        labels = ["median_income","housing_median_age","total_rooms","total_bedrooms","Population","households","Latitude","Longitude"]
        inputs = []

        for l in labels:
            inputs.append(st.number_input(l, 0.0))

        if st.button("Predecir"):
            X = np.array([inputs])
            Xs = scaler.transform(X)
            pred = modelo.predict(Xs)
            st.success(f"Precio estimado: ${pred[0]*100000:.2f}")


if __name__ == "__main__":
    main()


