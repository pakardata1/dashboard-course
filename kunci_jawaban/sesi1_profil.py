# Kunci Sesi 1 no.3 — aplikasi profil peserta
import streamlit as st

st.title("👤 Profil Peserta")
nama = st.text_input("Nama")
divisi = st.selectbox("Divisi", ["Finance", "Operations", "Marketing", "IT", "HR"])
pengalaman = st.slider("Pengalaman (tahun)", 0, 30, 2)
if nama:
    st.info(f"{nama} dari divisi {divisi} dengan pengalaman {pengalaman} tahun.")
