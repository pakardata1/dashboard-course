"""Membuat dataset latihan customer_churn.csv (data sintetis, reproducible)."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000
OUT = Path(__file__).parent / "customer_churn.csv"

region = rng.choice(["Jawa", "Sumatera", "Kalimantan", "Sulawesi", "Bali-Nusa Tenggara"],
                    N, p=[0.42, 0.22, 0.12, 0.14, 0.10])
segment = rng.choice(["Retail", "UMKM", "Korporat"], N, p=[0.60, 0.28, 0.12])
contract = rng.choice(["Bulanan", "Tahunan", "Dua Tahun"], N, p=[0.52, 0.30, 0.18])
payment = rng.choice(["Transfer Bank", "Kartu Kredit", "E-Wallet", "Virtual Account"],
                     N, p=[0.30, 0.22, 0.28, 0.20])
service = rng.choice(["Fiber", "DSL", "Mobile Broadband"], N, p=[0.48, 0.30, 0.22])

tenure = np.where(contract == "Bulanan", rng.integers(1, 36, N),
          np.where(contract == "Tahunan", rng.integers(6, 60, N), rng.integers(12, 72, N)))
base = {"Fiber": 420_000, "DSL": 260_000, "Mobile Broadband": 310_000}
seg_mult = {"Retail": 1.0, "UMKM": 1.35, "Korporat": 2.1}
monthly = np.array([base[s] * seg_mult[g] for s, g in zip(service, segment)])
monthly = np.round(monthly * rng.normal(1, 0.15, N), -3).clip(120_000)
total = np.round(monthly * tenure * rng.normal(1, 0.03, N), -3)
complaints = rng.poisson(np.where(service == "Fiber", 0.9, 1.3))
support_calls = rng.poisson(1.5 + 0.8 * complaints)

logit = (-1.6 + 1.5 * (contract == "Bulanan") - 0.6 * (contract == "Dua Tahun")
         - 0.035 * tenure + 0.0000022 * (monthly - 400_000)
         + 0.45 * complaints + 0.12 * support_calls
         + 0.35 * (payment == "Transfer Bank") - 0.25 * (segment == "Korporat"))
churn = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

start = np.datetime64("2023-01-01")
signup = start + rng.integers(0, 3 * 365, N).astype("timedelta64[D]")

df = pd.DataFrame({
    "customer_id": [f"CUST-{i:05d}" for i in range(1, N + 1)],
    "signup_date": pd.to_datetime(signup), "region": region, "segment": segment,
    "contract_type": contract, "payment_method": payment, "internet_service": service,
    "tenure_months": tenure, "monthly_charges": monthly, "total_charges": total,
    "num_complaints": complaints, "support_calls": support_calls, "churn": churn,
})
# Sedikit missing value agar ada latihan data quality di Sesi 2
miss = rng.choice(N, 45, replace=False)
df.loc[miss, "total_charges"] = np.nan
df.to_csv(OUT, index=False)
print(f"Tersimpan: {OUT}  shape={df.shape}  churn_rate={df.churn.mean():.3f}")
