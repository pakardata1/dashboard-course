# Kunci Sesi 2 no.1-4 (jalankan: python kunci_jawaban/sesi2_latihan.py)
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kunci_jawaban" / "output"; OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})
WARNA = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]

# 1a. Pie komposisi layanan internet
s = df["internet_service"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(s, labels=s.index, autopct="%1.1f%%", colors=WARNA[:3], startangle=90,
       wedgeprops=dict(edgecolor="white"))
fig.savefig(OUT / "1a_pie_layanan.png", dpi=150); plt.close(fig)

# 1b. Donut komposisi region, lalu 1c. bar horizontal terurut sebagai pembanding
r = df["region"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(r, labels=r.index, colors=WARNA, wedgeprops=dict(width=0.4, edgecolor="white"))
fig.savefig(OUT / "1b_donut_region.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(r.sort_values().index, r.sort_values().values, color=WARNA[0])
fig.savefig(OUT / "1c_barh_region.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2. Stacked bar 100% status per layanan internet
ct = pd.crosstab(df["internet_service"], df["status"], normalize="index") * 100
ax = ct.plot.barh(stacked=True, color=["#2a78d6", "#eb6834"], figsize=(6, 3))
ax.set_xlabel("Proporsi (%)"); ax.figure.savefig(OUT / "2_stacked_layanan.png", dpi=150,
                                                  bbox_inches="tight")
plt.close("all")

# 3. Box plot support_calls per status
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot([df.loc[df.status == s, "support_calls"] for s in ["Bertahan", "Churn"]],
           tick_labels=["Bertahan", "Churn"])
fig.savefig(OUT / "3_box_support_calls.png", dpi=150); plt.close(fig)

# 4. Line chart churn rate per kuartal pendaftaran
q = df.groupby(df.signup_date.dt.to_period("Q"))["churn"].mean() * 100
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(q.index.astype(str), q.values, marker="o", color="#eb6834")
ax.tick_params(axis="x", rotation=45); ax.set_ylabel("Churn rate (%)")
fig.savefig(OUT / "4_line_churn_kuartal.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print(ct.round(1)); print(q.round(1).head())
