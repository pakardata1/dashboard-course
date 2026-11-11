# Kunci Sesi 6 no.1-4 (jalankan: python kunci_jawaban/sesi6_latihan.py)
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kunci_jawaban" / "output"; OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
df["status"] = df["churn"].map({0: "Bertahan", 1: "Churn"})

# 1. Bar horizontal + garis rata-rata
pm = df.groupby("payment_method", as_index=False)["churn"].mean().sort_values("churn")
f1 = px.bar(pm, x="churn", y="payment_method", orientation="h", template="plotly_white",
            labels={"churn": "Churn rate", "payment_method": ""},
            title="Transfer Bank memiliki churn rate tertinggi")
f1.update_xaxes(tickformat=".0%")
f1.add_vline(x=df.churn.mean(), line_dash="dot", annotation_text="rata-rata")

# 2. Bubble chart
f2 = px.scatter(df, x="tenure_months", y="monthly_charges", color="status",
                size="support_calls", opacity=0.5, template="plotly_white")

# 3. Sunburst: lingkaran dalam = region, tengah = segmen, luar = kontrak; luas = jumlah pelanggan
cnt = df.groupby(["region", "segment", "contract_type"]).size().reset_index(name="pelanggan")
f3 = px.sunburst(cnt, path=["region", "segment", "contract_type"], values="pelanggan")

# 4. Gauge churn rate keseluruhan
f4 = go.Figure(go.Indicator(mode="gauge+number", value=df.churn.mean() * 100,
                            number={"suffix": "%"}, title={"text": "Churn rate"},
                            gauge={"axis": {"range": [0, 100]}}))
for i, f in enumerate([f1, f2, f3, f4], 1):
    f.write_html(OUT / f"latihan6_{i}.html", include_plotlyjs="cdn")
print("OK")
