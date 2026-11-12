# Jalankan dari root proyek:
# streamlit run sesi4/cek_inference.py
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.model_utils import load_metadata, load_model, predict, validate_input

st.set_page_config(
    page_title="Uji Inference Churn",
    page_icon="🧪",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metadata():
    return load_metadata()


st.title("🧪 Uji Inference Customer Churn")
st.caption("Uji model churn pada data pelanggan baru sebelum digunakan di dashboard.")

meta_path = ROOT / "models" / "model_metadata.json"
model_path = ROOT / "models" / "churn_model.joblib"
sample_path = ROOT / "data" / "new_customers.csv"

missing_files = [
    str(path.relative_to(ROOT))
    for path in (meta_path, model_path)
    if not path.is_file()
]
if missing_files:
    st.error(f"File model yang dibutuhkan tidak ditemukan: {', '.join(missing_files)}")
    st.stop()

meta = get_metadata()
model = get_model()

with st.expander("Informasi model"):
    st.write(f"**Model:** {meta['model_name']} · versi {meta['model_version']}")
    st.write(f"**Algoritma:** {meta['algorithm']}")
    st.write(f"**Fitur input:** {', '.join(meta['feature_order'])}")
    st.write(f"**Threshold keputusan:** {meta['decision_threshold']}")

uploaded_file = st.file_uploader(
    "Unggah data pelanggan baru (CSV)",
    type=["csv"],
)
use_sample = st.checkbox(
    "Gunakan contoh data/new_customers.csv",
    value=uploaded_file is None,
)

if uploaded_file is not None:
    customers = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
elif use_sample:
    if not sample_path.is_file():
        st.error("File contoh tidak ditemukan: data/new_customers.csv")
        st.stop()
    customers = pd.read_csv(sample_path)
    source_name = sample_path.name
else:
    st.info("Unggah file CSV atau pilih penggunaan dataset contoh.")
    st.stop()

st.caption(f"Sumber data: **{source_name}** · {len(customers):,} baris")

if customers.empty:
    st.warning("Data tidak berisi baris pelanggan untuk diprediksi.")
    st.stop()

validation_errors = validate_input(customers, meta)
if validation_errors:
    st.error("Data tidak sesuai dengan fitur yang dibutuhkan model:")
    for error in validation_errors:
        st.write(f"- {error}")
    st.stop()

results = predict(model, customers, meta)

total = len(results)
predicted_churn = int(results["predicted_churn"].sum())
average_probability = results["churn_probability"].mean()
high_risk = int(results["risk_category"].eq("Tinggi").sum())

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Pelanggan diproses", f"{total:,}")
metric2.metric("Prediksi churn", f"{predicted_churn:,}")
metric3.metric(
    "Rata-rata probabilitas churn",
    f"{average_probability:.1%}" if pd.notna(average_probability) else "N/A",
)
metric4.metric("Risiko tinggi", f"{high_risk:,}")

st.subheader("Distribusi kategori risiko")
risk_counts = (
    results["risk_category"]
    .value_counts()
    .reindex(["Rendah", "Sedang", "Tinggi"], fill_value=0)
    .rename_axis("Kategori risiko")
    .rename("Jumlah pelanggan")
)
st.bar_chart(risk_counts)

st.subheader("Hasil prediksi")
display_results = results.copy()
display_results["churn_probability"] *= 100
st.dataframe(
    display_results,
    width="stretch",
    hide_index=True,
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Probabilitas churn",
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        ),
    },
)
st.download_button(
    "Unduh hasil prediksi (CSV)",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="hasil_inference_churn.csv",
    mime="text/csv",
)
