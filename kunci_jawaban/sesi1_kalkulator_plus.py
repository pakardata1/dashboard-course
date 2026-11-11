# Kunci Sesi 1 no.1, 2, 4 — kalkulator dengan layanan Satelit, diskon promo, dan penghitung klik
import streamlit as st

st.title("🧾 Kalkulator Tagihan (versi latihan)")
HARGA = {"DSL": 260_000, "Fiber": 420_000, "Mobile Broadband": 310_000, "Satelit": 550_000}
DISKON = {"Bulanan": 0.00, "Tahunan": 0.10, "Dua Tahun": 0.18}

layanan = st.selectbox("Jenis layanan", list(HARGA))
kontrak = st.radio("Jenis kontrak", list(DISKON), horizontal=True)
promo = st.slider("Diskon promosi tambahan (%)", 0, 20, 0)
bulan = st.number_input("Durasi (bulan)", 1, 72, 12)

if "jumlah_klik" not in st.session_state:
    st.session_state.jumlah_klik = 0

if st.button("Hitung", type="primary"):
    st.session_state.jumlah_klik += 1
    per_bulan = HARGA[layanan] * (1 - DISKON[kontrak]) * (1 - promo / 100)
    st.metric("Biaya per bulan", f"Rp{per_bulan:,.0f}".replace(",", "."))
    st.metric("Total", f"Rp{per_bulan * bulan * 1.11:,.0f}".replace(",", "."))
st.caption(f"Tombol Hitung sudah diklik {st.session_state.jumlah_klik} kali.")
