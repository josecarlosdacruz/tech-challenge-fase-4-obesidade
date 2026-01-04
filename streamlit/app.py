import streamlit as st
import pandas as pd
import joblib

# Necessário para desserializar o preprocess.pkl
from custom_transformers import DropColumns, MinMax, OnHotEncodingNames

# Carregar artefatos de PRODUÇÃO
preprocess = joblib.load("preprocess.pkl")
model = joblib.load("model.pkl")

# DEBUG DEFINITIVO – REMOVER DEPOIS
st.write("Preprocess object:", preprocess)
st.write(
    "MinMax fitted:",
    hasattr(preprocess.named_steps['min_max'].scaler, 'min_')
)
st.write(
    "OneHot fitted:",
    hasattr(preprocess.named_steps['one_hot_enc'].encoder, 'categories_')
)


st.set_page_config(page_title="Predição de Obesidade", layout="centered")
st.title("🔍 Predição de Nível de Obesidade")

st.markdown("Preencha os dados abaixo:")

# ===== Inputs =====
Gender = st.selectbox("Gênero", ["Male", "Female"])
Age = st.number_input("Idade", min_value=14, max_value=80)
Height = st.number_input("Altura (cm)", min_value=100, max_value=230)
Weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0)

family_history = st.selectbox("Histórico familiar de obesidade?", ["yes", "no"])
FAVC = st.selectbox("Consome alimentos calóricos com frequência?", ["yes", "no"])
SMOKE = st.selectbox("Fuma?", ["yes", "no"])
SCC = st.selectbox("Monitora calorias?", ["yes", "no"])

FCVC = st.slider("Consumo de vegetais", 1.0, 3.0, 2.0)
NCP = st.slider("Número de refeições", 1.0, 4.0, 3.0)
CH2O = st.slider("Consumo de água", 1.0, 3.0, 2.0)
FAF = st.slider("Atividade física", 0.0, 3.0, 1.0)
TUE = st.slider("Uso de tecnologia", 0.0, 2.0, 1.0)

CAEC = st.selectbox("Come entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
CALC = st.selectbox("Consome álcool?", ["no", "Sometimes", "Frequently", "Always"])

MTRANS = st.selectbox(
    "Meio de transporte",
    ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
)

# ===== Botão =====
if st.button("🔮 Prever"):
    input_data = pd.DataFrame([{
        "Gender": 1 if Gender == "Female" else 0,
        "Age": int(Age),
        "Height": Height,
        "Weight": Weight,
        "family_history": 1 if family_history == "yes" else 0,
        "FAVC": 1 if FAVC == "yes" else 0,
        "SMOKE": 1 if SMOKE == "yes" else 0,
        "SCC": 1 if SCC == "yes" else 0,
        "FCVC": FCVC,
        "NCP": NCP,
        "CH2O": CH2O,
        "FAF": FAF,
        "TUE": TUE,
        "CAEC": CAEC,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    dict_frequencia = {
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    }

    input_data["Ind_CAEC"] = input_data["CAEC"].map(dict_frequencia)
    input_data["Ind_CALC"] = input_data["CALC"].map(dict_frequencia)

    # 🔒 TRANSFORM E PREDICT TOTALMENTE ISOLADOS
    processed = preprocess.transform(input_data)

    prediction = model.predict(processed)[0]
    proba = model.predict_proba(processed).max()

    st.success(f"🧠 Classificação prevista: **{prediction}**")
    st.info(f"Confiança do modelo: **{proba:.2%}**")
