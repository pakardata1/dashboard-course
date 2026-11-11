# Kunci Sesi 4 no.1, 3, 4 — threshold dinamis, crosstab, pendapatan berisiko
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, predict

@st.cache_resource
def get_model():
    return load_model()

meta = dict(load_metadata())
threshold = st.sidebar.slider("Threshold keputusan", 0.3, 0.7, 0.5, 0.05)
meta["decision_threshold"] = threshold          # salinan meta, file asli tidak berubah

data = pd.read_csv(ROOT / "data" / "new_customers.csv")
hasil = predict(get_model(), data, meta)

st.metric(f"Prediksi churn (threshold {threshold:.2f})", int(hasil.predicted_churn.sum()))
st.subheader("Pelanggan per region × kategori risiko")
st.dataframe(pd.crosstab(hasil.region, hasil.risk_category))
berisiko = hasil.loc[hasil.risk_category == "Tinggi", "monthly_charges"].sum()
st.metric("Pendapatan bulanan berisiko", f"Rp{berisiko:,.0f}".replace(",", "."))
