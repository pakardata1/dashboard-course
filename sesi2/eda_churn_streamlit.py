# Jalankan dari root proyek:
# streamlit run sesi2/eda_churn_streamlit.py
#
# Struktur folder yang diharapkan:
# proyek/
# ├── data/
# │   └── customer_churn.csv
# └── sesi2/
#     └── eda_churn_streamlit.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import FancyBboxPatch

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Warna visual
BIRU = "#2a78d6"
ORANYE = "#eb6834"
HIJAU = "#1baf7a"
KUNING = "#eda100"
UNGU = "#4a3aa7"
TEKS = "#0b0b0b"
TEKS2 = "#52514e"
GRID = "#e6e5e0"

STATUS = {"Bertahan": BIRU, "Churn": ORANYE}
KONTRAK = ["Bulanan", "Tahunan", "Dua Tahun"]

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#b9b8b2",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.labelcolor": TEKS2,
    "xtick.color": TEKS2,
    "ytick.color": TEKS2,
})

st.title("📊 Customer Churn Analytics")
st.caption(
    "Exploratory Data Analysis — profiling, cleaning, statistik deskriptif, "
    "segmentasi, dan visualisasi pelanggan."
)

# ============================================================
# 2. LOAD DATA
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "customer_churn.csv"

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if "signup_date" in data.columns:
        data["signup_date"] = pd.to_datetime(data["signup_date"], errors="coerce")
    return data

with st.sidebar:
    st.header("Pengaturan Data")
    st.write("Unggah dataset CSV atau gunakan dataset proyek.")
    uploaded_file = st.file_uploader("Upload customer churn CSV", type=["csv"])

try:
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        if "signup_date" in df_raw.columns:
            df_raw["signup_date"] = pd.to_datetime(
                df_raw["signup_date"], errors="coerce"
            )
        source_name = uploaded_file.name
    elif DEFAULT_DATA.exists():
        df_raw = load_data(str(DEFAULT_DATA))
        source_name = str(DEFAULT_DATA.name)
    else:
        st.error(
            "Dataset tidak ditemukan. Letakkan file customer_churn.csv pada "
            "folder data/ di direktori utama proyek, atau unggah melalui sidebar."
        )
        st.code(
            "proyek/\n"
            "├── data/\n"
            "│   └── customer_churn.csv\n"
            "└── sesi2/\n"
            "    └── eda_churn_streamlit.py",
            language="text",
        )
        st.stop()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()

# Validasi kolom wajib
required_cols = [
    "customer_id", "churn", "contract_type", "region", "segment",
    "internet_service", "payment_method", "tenure_months",
    "monthly_charges", "total_charges", "num_complaints",
    "support_calls", "signup_date",
]
missing_cols = [col for col in required_cols if col not in df_raw.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {', '.join(missing_cols)}")
    st.write("Kolom yang tersedia:", list(df_raw.columns))
    st.stop()

df = df_raw.copy()

# Normalisasi tipe data untuk analisis
for col in [
    "churn", "tenure_months", "monthly_charges", "total_charges",
    "num_complaints", "support_calls"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Status pelanggan
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())

# Kelompok tenure
df["kelompok_tenure"] = pd.cut(
    df["tenure_months"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"],
    include_lowest=True,
)

num_cols = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_complaints", "support_calls"
]

# ============================================================
# 3. SIDEBAR FILTER
# ============================================================
with st.sidebar:
    st.divider()
    st.header("Filter Data")

    regions = sorted(df["region"].dropna().astype(str).unique().tolist())
    selected_regions = st.multiselect(
        "Region", regions, default=regions
    )

    contracts = sorted(df["contract_type"].dropna().astype(str).unique().tolist())
    selected_contracts = st.multiselect(
        "Jenis kontrak", contracts, default=contracts
    )

    segments = sorted(df["segment"].dropna().astype(str).unique().tolist())
    selected_segments = st.multiselect(
        "Segmen pelanggan", segments, default=segments
    )

    statuses = ["Bertahan", "Churn"]
    selected_status = st.multiselect(
        "Status pelanggan", statuses, default=statuses
    )

filtered = df[
    df["region"].astype(str).isin(selected_regions)
    & df["contract_type"].astype(str).isin(selected_contracts)
    & df["segment"].astype(str).isin(selected_segments)
    & df["status"].isin(selected_status)
].copy()

if filtered.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter.")
    st.stop()

# ============================================================
# 4. HELPER
# ============================================================
def show_fig(fig, width="stretch"):
    """Menampilkan Matplotlib figure di Streamlit lalu menutupnya."""
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def format_rupiah_ribu(value):
    return f"Rp{value / 1000:,.0f} rb".replace(",", ".")


def section_title(title, description=None):
    st.subheader(title)
    if description:
        st.caption(description)


# ============================================================
# 5. KPI OVERVIEW
# ============================================================
total_pelanggan = len(filtered)
churn_rate = filtered["churn"].mean()
jumlah_churn = int(filtered["churn"].sum())
rata_tagihan = filtered["monthly_charges"].mean()
rata_tenure = filtered["tenure_months"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total pelanggan", f"{total_pelanggan:,}".replace(",", "."))
k2.metric(
    "Churn rate",
    f"{churn_rate:.1%}" if pd.notna(churn_rate) else "N/A",
    help="Proporsi pelanggan dengan churn = 1.",
)
k3.metric(
    "Rata-rata tagihan",
    format_rupiah_ribu(rata_tagihan) if pd.notna(rata_tagihan) else "N/A",
)
k4.metric(
    "Rata-rata tenure",
    f"{rata_tenure:.1f} bulan" if pd.notna(rata_tenure) else "N/A",
)

st.caption(f"Dataset: **{source_name}** | Menampilkan **{len(filtered):,}** dari **{len(df):,}** baris setelah filter.")
st.divider()

# ============================================================
# 6. TAB NAVIGASI
# ============================================================
tabs = st.tabs([
    "Overview",
    "Data Profiling",
    "Statistik & Cleaning",
    "Filtering & Grouping",
    "Visualisasi EDA",
    "Business Insights",
])

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tabs[0]:
    section_title("Ringkasan Status Pelanggan", "Proporsi pelanggan bertahan dan churn berdasarkan filter aktif.")

    c1, c2 = st.columns(2)

    with c1:
        counts = filtered["status"].value_counts().reindex(
            ["Bertahan", "Churn"], fill_value=0
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            counts,
            labels=[
                f"{name}\n{value:,} pelanggan".replace(",", ".")
                for name, value in counts.items()
            ],
            colors=[STATUS[name] for name in counts.index],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            pctdistance=0.68,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(color=TEKS),
        )
        for t in ax.texts:
            if "%" in t.get_text():
                t.set_color("white")
                t.set_fontweight("bold")
        ax.set_title("Komposisi Status Pelanggan")
        show_fig(fig)

    with c2:
        contract_counts = (
            filtered["contract_type"].value_counts()
            .reindex(KONTRAK)
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        if contract_counts.sum() > 0:
            ax.pie(
                contract_counts,
                colors=[BIRU, HIJAU, KUNING][:len(contract_counts)],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
            )
            ax.text(
                0, 0.06, f"{len(filtered):,}".replace(",", "."),
                ha="center", fontsize=18, fontweight="bold"
            )
            ax.text(0, -0.16, "pelanggan", ha="center", color=TEKS2)
            ax.legend(
                [
                    f"{name}: {value / len(filtered):.1%}"
                    for name, value in contract_counts.items()
                ],
                loc="center left",
                bbox_to_anchor=(0.95, 0.5),
                frameon=False,
            )
        ax.set_title("Komposisi Jenis Kontrak")
        show_fig(fig)

    st.markdown("#### Cuplikan Dataset")
    st.dataframe(filtered.head(10), width="stretch")

# ------------------------------------------------------------
# TAB 2: DATA PROFILING
# ------------------------------------------------------------
with tabs[1]:
    section_title("Data Profiling", "Pemeriksaan struktur, tipe data, missing values, nilai unik, dan duplikasi.")

    profiling = pd.DataFrame({
        "tipe_data": filtered.dtypes.astype(str),
        "missing": filtered.isna().sum(),
        "missing_%": (filtered.isna().mean() * 100).round(2),
        "nilai_unik": filtered.nunique(dropna=True),
    })
    st.dataframe(profiling, width="stretch")

    p1, p2, p3 = st.columns(3)
    p1.metric("Jumlah baris", f"{len(filtered):,}".replace(",", "."))
    p2.metric("Jumlah kolom", f"{filtered.shape[1]:,}".replace(",", "."))
    p3.metric(
        "Duplikasi customer_id",
        f"{filtered['customer_id'].duplicated().sum():,}".replace(",", "."),
    )

    st.markdown("#### Ringkasan Missing Values")
    missing_summary = (
        filtered.isna().sum()
        .rename("jumlah_missing")
        .to_frame()
        .query("jumlah_missing > 0")
        .sort_values("jumlah_missing", ascending=False)
    )
    if missing_summary.empty:
        st.success("Tidak ditemukan missing values pada data hasil filter.")
    else:
        st.dataframe(missing_summary, width="stretch")

# ------------------------------------------------------------
# TAB 3: STATISTIK & CLEANING
# ------------------------------------------------------------
with tabs[2]:
    section_title("Statistik Deskriptif", "Ukuran pemusatan, penyebaran, dan kuartil untuk variabel numerik.")
    st.dataframe(filtered[num_cols].describe().T.round(2), width="stretch")

    section_title("Cleaning Ringan", "Pada analisis ini, missing value total_charges diisi menggunakan median.")
    cleaning = pd.DataFrame({
        "indikator": ["Missing total_charges sebelum imputasi", "Missing total_charges setelah imputasi"],
        "jumlah": [
            int(df_raw["total_charges"].isna().sum()),
            int(df["total_charges"].isna().sum()),
        ],
    })
    st.dataframe(cleaning, width="stretch")

    st.markdown("#### Deteksi Outlier dengan IQR")
    outlier_rows = []
    for col in num_cols:
        series = filtered[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        outlier_rows.append({
            "variabel": col,
            "batas_bawah": round(lower, 2),
            "batas_atas": round(upper, 2),
            "jumlah_outlier": count,
        })
    st.dataframe(pd.DataFrame(outlier_rows), width="stretch")

# ------------------------------------------------------------
# TAB 4: FILTERING & GROUPING
# ------------------------------------------------------------
with tabs[3]:
    section_title("Contoh Filtering", "Pelanggan dengan kontrak bulanan dan jumlah keluhan minimal tiga.")
    berisiko = filtered[
        (filtered["contract_type"] == "Bulanan")
        & (filtered["num_complaints"] >= 3)
    ]
    st.metric("Pelanggan sesuai kriteria", f"{len(berisiko):,}".replace(",", "."))
    if not berisiko.empty:
        risk_churn = berisiko["churn"].mean()
        st.metric("Churn rate kelompok tersebut", f"{risk_churn:.1%}")
        st.dataframe(berisiko.head(20), width="stretch")
    else:
        st.info("Tidak ada pelanggan yang memenuhi kriteria pada filter aktif.")

    st.markdown("#### Churn berdasarkan Jenis Kontrak")
    churn_kontrak = (
        filtered.groupby("contract_type")
        .agg(
            pelanggan=("customer_id", "count"),
            churn_rate=("churn", "mean"),
            rata2_tagihan=("monthly_charges", "mean"),
        )
        .reindex(KONTRAK)
    )
    st.dataframe(churn_kontrak.round(3), width="stretch")

    st.markdown("#### Pivot Table: Region × Segmen")
    pivot = pd.pivot_table(
        filtered,
        values="churn",
        index="region",
        columns="segment",
        aggfunc="mean",
    )
    st.dataframe((pivot * 100).round(1), width="stretch")

# ------------------------------------------------------------
# TAB 5: VISUALISASI EDA
# ------------------------------------------------------------
with tabs[4]:
    st.info("Seluruh grafik di bawah mengikuti filter Region, Jenis Kontrak, Segmen, dan Status pada sidebar.")

    # 7.1 Bar churn per kontrak
    section_title("1. Churn Rate berdasarkan Jenis Kontrak")
    churn_kontrak_plot = (
        filtered.groupby("contract_type")["churn"].mean()
        .reindex(KONTRAK).dropna() * 100
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(churn_kontrak_plot.index, churn_kontrak_plot.values, color=ORANYE, width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=TEKS)
    ax.axhline(filtered["churn"].mean() * 100, color=TEKS2, linestyle="--", linewidth=1)
    ax.set(ylabel="Churn rate (%)")
    ax.set_title("Churn Rate per Jenis Kontrak")
    show_fig(fig)

    # 7.2 Churn per region
    section_title("2. Churn Rate berdasarkan Region")
    reg = filtered.groupby("region")["churn"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(reg.index, reg.values, color=BIRU, height=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=TEKS)
    ax.set(xlabel="Churn rate (%)")
    ax.set_title("Perbandingan Churn antar-Region")
    show_fig(fig)

    # 7.3 Grouped bar segment x kontrak
    section_title("3. Churn Rate: Segmen × Jenis Kontrak")
    gb = (
        filtered.pivot_table(
            values="churn", index="contract_type",
            columns="segment", aggfunc="mean"
        ).reindex(KONTRAK) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not gb.empty:
        x = np.arange(len(gb.index))
        width = 0.8 / max(len(gb.columns), 1)
        colors = [BIRU, ORANYE, HIJAU, UNGU, KUNING]
        for i, seg in enumerate(gb.columns):
            offset = (i - (len(gb.columns) - 1) / 2) * width
            ax.bar(x + offset, gb[seg], width=width * 0.9,
                   color=colors[i % len(colors)], label=str(seg))
        ax.set_xticks(x, gb.index)
        ax.set_ylabel("Churn rate (%)")
        ax.legend(title="Segmen", frameon=False)
    ax.set_title("Pola Churn berdasarkan Segmen dan Kontrak")
    show_fig(fig)

    # 7.4 Stacked bar pembayaran
    section_title("4. Komposisi Status berdasarkan Metode Pembayaran")
    sb = pd.crosstab(
        filtered["payment_method"], filtered["status"], normalize="index"
    ) * 100
    for status in ["Bertahan", "Churn"]:
        if status not in sb.columns:
            sb[status] = 0
    sb = sb[["Bertahan", "Churn"]].sort_values("Churn")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(sb.index, sb["Bertahan"], color=BIRU, label="Bertahan", height=0.55)
    ax.barh(sb.index, sb["Churn"], left=sb["Bertahan"], color=ORANYE, label="Churn", height=0.55)
    for i, value in enumerate(sb["Churn"]):
        if value > 5:
            ax.text(100 - value / 2, i, f"{value:.1f}%", ha="center", va="center",
                    color="white", fontweight="bold")
    ax.set(xlabel="Proporsi pelanggan (%)", xlim=(0, 100))
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_title("Status Pelanggan per Metode Pembayaran")
    show_fig(fig)

    # 7.5 Histogram tenure
    section_title("5. Distribusi Tenure berdasarkan Status")
    fig, ax = plt.subplots(figsize=(8, 4))
    for status in ["Bertahan", "Churn"]:
        values = filtered.loc[filtered["status"] == status, "tenure_months"].dropna()
        if not values.empty:
            ax.hist(values, bins=24, alpha=0.60, color=STATUS[status], label=status,
                    edgecolor="white")
    ax.set(xlabel="Tenure (bulan)", ylabel="Jumlah pelanggan")
    ax.legend(frameon=False)
    ax.set_title("Distribusi Tenure Pelanggan")
    show_fig(fig)

    # 7.6 Boxplot tagihan
    section_title("6. Distribusi Tagihan Bulanan berdasarkan Status")
    box_data, box_labels = [], []
    for status in ["Bertahan", "Churn"]:
        values = (filtered.loc[filtered["status"] == status, "monthly_charges"] / 1000).dropna()
        if not values.empty:
            box_data.append(values)
            box_labels.append(status)
    fig, ax = plt.subplots(figsize=(7, 4))
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5)
        for patch, status in zip(bp["boxes"], box_labels):
            patch.set_facecolor(STATUS[status])
            patch.set_alpha(0.55)
    ax.set(ylabel="Tagihan bulanan (ribu Rp)")
    ax.set_title("Perbandingan Tagihan Bulanan")
    show_fig(fig)

    # 7.7 Violin tenure per kontrak
    section_title("7. Distribusi Tenure berdasarkan Jenis Kontrak")
    violin_data, violin_labels = [], []
    for contract in KONTRAK:
        values = filtered.loc[
            filtered["contract_type"] == contract, "tenure_months"
        ].dropna()
        if len(values) > 1:
            violin_data.append(values)
            violin_labels.append(contract)
    fig, ax = plt.subplots(figsize=(7, 4))
    if violin_data:
        vp = ax.violinplot(violin_data, showmedians=True)
        for body, color in zip(vp["bodies"], [BIRU, HIJAU, KUNING]):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(violin_labels) + 1), violin_labels)
    ax.set_ylabel("Tenure (bulan)")
    ax.set_title("Sebaran Tenure per Jenis Kontrak")
    show_fig(fig)

    # 7.8 Line signup per bulan
    section_title("8. Tren Pelanggan Baru per Bulan")
    date_data = filtered.dropna(subset=["signup_date"]).copy()
    if not date_data.empty:
        bulanan = date_data.set_index("signup_date").resample("ME")["customer_id"].count()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(bulanan.index, bulanan.values, color="#9dbbe6", linewidth=1.5,
                marker="o", markersize=3, label="Per bulan")
        ax.plot(bulanan.index, bulanan.rolling(3).mean(), color=BIRU,
                linewidth=2.2, label="Rata-rata bergerak 3 bulan")
        ax.set(ylabel="Pelanggan baru", ylim=(0, None))
        ax.legend(frameon=False)
        ax.set_title("Tren Akuisisi Pelanggan")
        show_fig(fig)
    else:
        st.warning("Tidak ada tanggal signup_date yang valid.")

    # 7.9 Churn per kelompok tenure
    section_title("9. Churn Rate berdasarkan Kelompok Tenure")
    kt = (
        filtered.groupby("kelompok_tenure", observed=True)["churn"]
        .mean() * 100
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    if not kt.empty:
        ax.plot(kt.index.astype(str), kt.values, color=ORANYE,
                marker="o", markersize=8, linewidth=2)
        for x_label, y_value in zip(kt.index.astype(str), kt.values):
            ax.annotate(f"{y_value:.1f}%", (x_label, y_value),
                        textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set(xlabel="Kelompok tenure (bulan)", ylabel="Churn rate (%)")
    ax.set_title("Churn Rate per Kelompok Tenure")
    show_fig(fig)

    # 7.10 Stacked area per kuartal
    section_title("10. Akuisisi Pelanggan per Kuartal dan Segmen")
    date_data = filtered.dropna(subset=["signup_date"]).copy()
    if not date_data.empty:
        area = (
            date_data.assign(
                kuartal=date_data["signup_date"].dt.to_period("Q").dt.to_timestamp()
            )
            .pivot_table(
                index="kuartal", columns="segment",
                values="customer_id", aggfunc="count"
            )
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        if not area.empty:
            ax.stackplot(
                area.index, area.T.values, labels=area.columns,
                colors=[BIRU, ORANYE, HIJAU, UNGU, KUNING][:len(area.columns)],
                alpha=0.85, edgecolor="white", linewidth=1
            )
            ax.legend(loc="upper left", ncol=3, frameon=False)
        ax.set_ylabel("Pelanggan baru per kuartal")
        ax.set_title("Komposisi Akuisisi per Segmen")
        show_fig(fig)

    # 7.11 Scatter tenure vs charges
    section_title("11. Hubungan Tenure dan Tagihan Bulanan")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for status in ["Bertahan", "Churn"]:
        d = filtered[filtered["status"] == status]
        ax.scatter(
            d["tenure_months"], d["monthly_charges"] / 1000,
            s=12, alpha=0.5, color=STATUS[status], label=status
        )
    ax.set(xlabel="Tenure (bulan)", ylabel="Tagihan bulanan (ribu Rp)")
    ax.legend(frameon=False)
    ax.set_title("Scatter Plot Tenure vs Tagihan")
    show_fig(fig)

    # 7.12 Heatmap region x segment
    section_title("12. Heatmap Churn Rate: Region × Segmen")
    pivot_heat = pd.pivot_table(
        filtered, values="churn", index="region",
        columns="segment", aggfunc="mean"
    ) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not pivot_heat.empty:
        im = ax.imshow(pivot_heat.values, cmap="Oranges", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot_heat.columns)), pivot_heat.columns)
        ax.set_yticks(range(len(pivot_heat.index)), pivot_heat.index)
        ax.grid(False)
        for i in range(pivot_heat.shape[0]):
            for j in range(pivot_heat.shape[1]):
                value = pivot_heat.values[i, j]
                if pd.notna(value):
                    ax.text(
                        j, i, f"{value:.1f}%", ha="center", va="center",
                        color="white" if value > 55 else TEKS
                    )
        fig.colorbar(im, ax=ax, label="Churn rate (%)")
    ax.set_title("Churn Rate per Region dan Segmen")
    show_fig(fig)

    # 7.13 Correlation heatmap
    section_title("13. Heatmap Korelasi Variabel Numerik")
    corr_cols = num_cols + ["churn"]
    corr = filtered[corr_cols].corr()
    labels = [
        "Tenure", "Tagihan", "Total", "Keluhan", "Panggilan CS", "Churn"
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            if pd.notna(value):
                ax.text(
                    j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else TEKS, fontsize=8
                )
    fig.colorbar(im, ax=ax, label="Korelasi Pearson")
    ax.set_title("Korelasi antarvariabel")
    show_fig(fig)

# ------------------------------------------------------------
# TAB 6: BUSINESS INSIGHTS
# ------------------------------------------------------------
with tabs[5]:
    section_title("Business Insights", "Ringkasan otomatis berdasarkan data yang sedang ditampilkan.")

    st.markdown("#### 1. Churn berdasarkan kontrak")
    contract_churn = (
        filtered.groupby("contract_type")["churn"].mean().dropna().sort_values(ascending=False)
    )
    if not contract_churn.empty:
        highest_contract = contract_churn.index[0]
        st.write(
            f"Jenis kontrak dengan churn rate tertinggi adalah **{highest_contract}** "
            f"({contract_churn.iloc[0]:.1%})."
        )
    else:
        st.write("Belum cukup data untuk membandingkan churn berdasarkan kontrak.")

    st.markdown("#### 2. Churn berdasarkan region")
    region_churn = (
        filtered.groupby("region")["churn"].mean().dropna().sort_values(ascending=False)
    )
    if not region_churn.empty:
        st.write(
            f"Region dengan churn rate tertinggi adalah **{region_churn.index[0]}** "
            f"({region_churn.iloc[0]:.1%})."
        )
    else:
        st.write("Belum cukup data untuk membandingkan churn berdasarkan region.")

    st.markdown("#### 3. Hubungan tenure dengan churn")
    tenure_churn = (
        filtered.groupby("kelompok_tenure", observed=True)["churn"]
        .mean().dropna().sort_values(ascending=False)
    )
    if not tenure_churn.empty:
        group_name = str(tenure_churn.index[0])
        st.write(
            f"Kelompok tenure dengan churn rate tertinggi adalah **{group_name} bulan** "
            f"({tenure_churn.iloc[0]:.1%})."
        )
    else:
        st.write("Belum cukup data untuk menganalisis churn berdasarkan tenure.")

    st.warning(
        "Catatan interpretasi: hasil EDA menunjukkan pola dan asosiasi dalam data, "
        "bukan bukti hubungan sebab-akibat."
    )

# ============================================================
# 7. FOOTER
# ============================================================
st.divider()
st.caption("Customer Churn Analytics | Exploratory Data Analysis")
