# .venv/bin/python sesi8/app_final_dashboard.py — FINAL DASHBOARD PROJECT (Dash + Plotly + callback + ML inference)
# Struktur: Sidebar navigasi dengan 5 halaman
#   1. Data Overview   2. Exploratory Data Analysis   3. Prediksi Single Observasi
#   4. Prediksi Multi Observasi (upload CSV)   5. Tim Developer

import sys
import io
import base64
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, ctx, ALL
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from utils.model_utils import load_model, load_metadata, validate_input, predict

# ============ 1. DATA + MODEL (dimuat sekali saat aplikasi start) ============
META = load_metadata()
MODEL = load_model()
raw = pd.read_csv(ROOT / "data" / "customer_churn.csv", parse_dates=["signup_date"])
RAW_MISSING = raw.isna().sum()                       # simpan info missing value sebelum imputasi
raw["total_charges"] = raw["total_charges"].fillna(raw["monthly_charges"] * raw["tenure_months"])
DF = predict(MODEL, raw, META)                      # batch inference seluruh pelanggan
DF["status"] = DF["churn"].map({0: "Bertahan", 1: "Churn"})
DF["periode"] = DF["signup_date"].dt.to_period("Q").astype(str)
RISK_ORDER = ["Rendah", "Sedang", "Tinggi"]
RISK_COLOR = {"Rendah": "#55A868", "Sedang": "#DD8452", "Tinggi": "#C44E52"}
STATUS_COLOR = {"Bertahan": "#4C72B0", "Churn": "#C44E52"}
CATS = META["categories"]

NUM_COLS = ["tenure_months", "monthly_charges", "total_charges", "num_complaints", "support_calls"]
CAT_COLS = ["region", "segment", "contract_type", "payment_method", "internet_service"]
FEATURES = NUM_COLS + CAT_COLS                       # kolom wajib untuk prediksi
COL_LABEL = {"tenure_months": "Tenure (bulan)", "monthly_charges": "Tagihan bulanan",
             "total_charges": "Total tagihan", "num_complaints": "Jumlah keluhan",
             "support_calls": "Panggilan CS", "region": "Region", "segment": "Segmen",
             "contract_type": "Jenis kontrak", "payment_method": "Metode pembayaran",
             "internet_service": "Layanan internet"}
CAT_ORDERS = {"risk_category": RISK_ORDER, "contract_type": ["Bulanan", "Tahunan", "Dua Tahun"],
              "status": ["Bertahan", "Churn"]}

# Anggota tim — silakan ganti dengan data tim Anda
TEAM = [
    {"nama": "Nama Anggota 1", "peran": "Project Lead & Data Scientist",
     "tugas": "Perumusan masalah bisnis, pemodelan, dan evaluasi model", "email": "anggota1@email.com"},
    {"nama": "Nama Anggota 2", "peran": "Data Engineer",
     "tugas": "Pembersihan data, feature engineering, dan pipeline inferensi", "email": "anggota2@email.com"},
    {"nama": "Nama Anggota 3", "peran": "Data Analyst",
     "tugas": "Exploratory data analysis dan penyusunan insight bisnis", "email": "anggota3@email.com"},
    {"nama": "Nama Anggota 4", "peran": "Dashboard Developer",
     "tugas": "Desain antarmuka, callback Dash, dan deployment", "email": "anggota4@email.com"},
]

# ============ 2. STYLE ============
C = {"bg": "#F5F7FB", "card": "white", "ink": "#1F2A44", "muted": "#6B7280", "accent": "#2F5DA8",
     "side": "#1F2A44", "side_ink": "#C9D2E3"}
CARD = {"background": C["card"], "borderRadius": "10px", "padding": "10px 12px", "minWidth": 0,
        "boxShadow": "0 1px 3px rgba(0,0,0,.08)", "boxSizing": "border-box"}
# minmax(0, 1fr) mencegah chart/tabel mendorong kolom grid melebihi lebar layar
GRID2 = {"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "10px"}
GRID3 = {"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "10px"}
H_CHART = 300                                         # tinggi standar chart (px) agar rapat tanpa ruang kosong


GRAPH_IDS = []                                        # semua chart yang bisa dibuka sebagai pop up


def graph(id_, h=H_CHART, **kw):
    """Chart + tombol ⤢ di pojok kanan atas untuk membuka chart dalam pop up."""
    GRAPH_IDS.append(id_)
    return html.Div(className="chart-wrap", style={"position": "relative"}, children=[
        html.Button("⤢", id={"type": "expand", "index": id_}, n_clicks=0, className="expand-btn",
                    title="Perbesar chart (pop up)"),
        dcc.Graph(id=id_, style={"height": f"{h}px"}, config={"displaylogo": False, "responsive": True}, **kw)])


def tidy(fig, legend_top=True):
    """Rapikan margin & judul semua figure supaya mengisi kartu tanpa ruang kosong."""
    fig.update_layout(margin=dict(l=8, r=8, t=40, b=8), title_font_size=14, title_x=0.01,
                      font_size=11, autosize=True)
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                                      title_text=""))
    return fig
BTN = {"background": C["accent"], "color": "white", "border": 0, "padding": "10px 18px",
       "borderRadius": "8px", "cursor": "pointer", "fontWeight": 600}
BTN_OUTLINE = {**BTN, "background": "white", "color": C["accent"], "border": f"1px solid {C['accent']}"}
TABLE_STYLE = dict(style_cell={"padding": "4px 6px", "fontSize": "12px", "textAlign": "left",
                               "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
                               "maxWidth": "220px"},
                   style_header={"fontWeight": "bold", "background": "#E9EDF5"},
                   style_table={"overflowX": "auto"})
RISK_COND = [
    {"if": {"filter_query": "{risk_category} = 'Tinggi'", "column_id": "risk_category"},
     "color": "white", "backgroundColor": RISK_COLOR["Tinggi"]},
    {"if": {"filter_query": "{risk_category} = 'Sedang'", "column_id": "risk_category"},
     "backgroundColor": "#FBE3D3"}]


def label(t): return html.Label(t, style={"fontWeight": 600, "fontSize": "13px"})
def kpi(judul, nilai, sub=""):
    return html.Div(style={**CARD, "flex": 1}, children=[
        html.Div(judul, style={"color": C["muted"], "fontSize": "12px"}),
        html.Div(nilai, style={"fontSize": "22px", "fontWeight": 700, "color": C["ink"],
                               "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
        html.Div(sub, style={"color": C["muted"], "fontSize": "11px"})])
KPI_ROW = {"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "10px",
           "marginBottom": "10px"}
def rupiah(x): return f"Rp{x:,.0f}".replace(",", ".")
def card(*children, **style): return html.Div(style={**CARD, **style}, children=list(children))
def page_header(judul, deskripsi):
    return html.Div(style={"marginBottom": "10px"}, children=[
        html.H2(judul, style={"margin": 0, "fontSize": "20px"}),
        html.Div(deskripsi, style={"color": C["muted"], "fontSize": "13px"})])
def section(t): return html.H3(t, style={"margin": "12px 0 8px", "fontSize": "15px"})
def empty_fig(msg="Tidak ada data — longgarkan filter"):
    f = go.Figure()
    f.add_annotation(text=msg, showarrow=False, font_size=14, font_color=C["muted"])
    f.update_layout(template="plotly_white", xaxis_visible=False, yaxis_visible=False,
                    margin=dict(l=8, r=8, t=8, b=8))
    return f
def risk_columns(cols):
    return [{"name": n, "id": i, **({"type": "numeric", "format": {"specifier": ".1%"}}
                                    if i == "churn_probability" else {})} for n, i in cols]


# ============ 3. SIDEBAR ============
PAGES = [("overview", "/", "📊", "Data Overview"),
         ("eda", "/eda", "🔍", "Exploratory Data Analysis"),
         ("single", "/prediksi-single", "👤", "Prediksi Single Observasi"),
         ("batch", "/prediksi-multi", "📁", "Prediksi Multi Observasi"),
         ("team", "/tim", "👥", "Tim Developer")]
PATH_TO_PAGE = {path: key for key, path, _, _ in PAGES}
NAV_BASE = {"display": "flex", "alignItems": "center", "gap": "10px", "padding": "8px 10px",
            "borderRadius": "8px", "textDecoration": "none", "color": C["side_ink"], "fontSize": "13px",
            "marginBottom": "2px", "whiteSpace": "nowrap", "overflow": "hidden"}
NAV_ACTIVE = {**NAV_BASE, "background": C["accent"], "color": "white", "fontWeight": 600}

side_label = lambda t: html.Div(t, style={"fontWeight": 600, "fontSize": "12px", "color": "white",
                                          "margin": "10px 0 4px"})
# RangeSlider Dash 4 menampilkan kotak input min/max; dimatikan agar muat di sidebar sempit
_slider_kw = ({"allow_direct_input": False}
              if "allow_direct_input" in dcc.RangeSlider().available_properties else {})
filter_panel = html.Div(id="filter-panel", children=[
    html.Hr(style={"borderColor": "#3A4766", "margin": "12px 0 6px"}),
    html.Div("FILTER DATA", style={"fontSize": "10px", "letterSpacing": "1px", "color": "#8A96B0"}),
    side_label("Region"), dcc.Dropdown(id="f-region", options=CATS["region"], multi=True, placeholder="Semua"),
    side_label("Segmen"),
    dcc.Dropdown(id="f-segment", options=CATS["segment"], multi=True, placeholder="Semua"),
    side_label("Jenis kontrak"),
    dcc.Checklist(id="f-kontrak", options=CATS["contract_type"], value=CATS["contract_type"],
                  className="side-check", labelStyle={"color": "white"}),
    side_label("Kategori risiko (prediksi)"),
    dcc.Checklist(id="f-risk", options=RISK_ORDER, value=RISK_ORDER, inline=True,
                  className="side-check", labelStyle={"color": "white"}),
    side_label("Tenure (bulan)"),
    dcc.RangeSlider(id="f-tenure", min=1, max=72, value=[1, 72], step=1, className="side-slider",
                    tooltip={"placement": "bottom", "always_visible": False},
                    marks={i: {"label": str(i), "style": {"color": "white"}} for i in (1, 24, 48, 72)},
                    **_slider_kw),
])

sidebar = html.Div(id="sidebar", className="sidebar", children=[
    html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between",
                    "marginBottom": "14px", "gap": "6px"}, children=[
        html.Div(className="side-hide", children=[
            html.Div("Customer Churn", style={"fontSize": "16px", "fontWeight": 700, "whiteSpace": "nowrap"}),
            html.Div("Intelligence Dashboard", style={"fontSize": "11px", "color": C["side_ink"],
                                                      "whiteSpace": "nowrap"})]),
        html.Button("☰", id="btn-toggle", n_clicks=0, title="Perkecil / perbesar sidebar",
                    className="toggle-btn")]),
    html.Div([dcc.Link([html.Span(icon, className="nav-icon"), html.Span(nama, className="side-hide")],
                       href=path, id=f"nav-{key}", style=NAV_BASE, title=nama)
              for key, path, icon, nama in PAGES]),
    html.Div(className="side-hide", children=[
        filter_panel,
        html.Hr(style={"borderColor": "#3A4766", "margin": "12px 0 6px"}),
        html.Small(f"Model: {META['model_name']} v{META['model_version']}", style={"color": "#8A96B0"})]),
])

# ============ 4. HALAMAN 1 — DATA OVERVIEW ============
page_overview = html.Div(id="page-overview", children=[
    page_header("Data Overview",
                "Pertanyaan bisnis: pelanggan mana yang berisiko churn dan apa yang harus dilakukan? "
                "· Data latihan 2023–2025 · Prediksi dari final trained model"),
    html.Div(id="kpi-row", style=KPI_ROW),
    html.Div(style={**GRID2, "gridTemplateColumns": "minmax(0, 2fr) minmax(0, 3fr)"}, children=[
        card(graph("g-status", 280)),
        card(html.H4("Informasi dataset", style={"margin": "0 0 6px", "fontSize": "14px"}),
             html.Div(id="dataset-info"))]),
    html.Div(style={**GRID2, "marginTop": "10px"}, children=[
        card(html.H4("Profil kolom", style={"margin": "0 0 2px", "fontSize": "14px"}),
             html.Small("Missing dihitung dari data mentah sebelum imputasi total_charges.",
                        style={"color": C["muted"]}),
             dash_table.DataTable(id="tbl-profile", page_size=7, **TABLE_STYLE)),
        card(html.H4("Statistik deskriptif variabel numerik", style={"margin": "0 0 6px", "fontSize": "14px"}),
             dash_table.DataTable(id="tbl-describe", **TABLE_STYLE))]),
    card(html.H4("Pratinjau data", style={"margin": "0 0 6px", "fontSize": "14px"}),
         dash_table.DataTable(id="tbl-preview", page_size=8, sort_action="native",
                              filter_action="native", **TABLE_STYLE), marginTop="10px"),
])

# ============ 5. HALAMAN 2 — EDA ============
page_eda = html.Div(id="page-eda", children=[
    page_header("Exploratory Data Analysis",
                "Pola churn aktual dan hasil prediksi model berdasarkan filter di sidebar"),
    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "10px"},
             children=[
        label("Variabel numerik:"),
        dcc.Dropdown(id="eda-num", options=[{"label": COL_LABEL[c], "value": c} for c in NUM_COLS],
                     value="tenure_months", clearable=False, style={"width": "200px"}),
        label("Variabel kategorik:"),
        dcc.Dropdown(id="eda-cat", options=[{"label": COL_LABEL[c], "value": c} for c in CAT_COLS],
                     value="contract_type", clearable=False, style={"width": "200px"})]),
    html.Div(style=GRID2, children=[
        card(graph("g-num-dist")),
        card(graph("g-cat-rate")),
        card(graph("g-box")),
        card(graph("g-corr"))]),
    section("Analisis risiko (hasil prediksi)"),
    html.Div(style=GRID3, children=[
        card(graph("g-donut", 280)),
        card(graph("g-region-risk", 280)),
        card(graph("g-contract", 280)),
        card(graph("g-trend", 280)),
        card(graph("g-risk", 280)),
        card(graph("g-actual-pred", 280))]),
    card(html.H4(id="tbl-title", style={"margin": "0 0 6px", "fontSize": "14px"}),
         dash_table.DataTable(
             id="tbl-risk", page_size=10, sort_action="native", filter_action="native",
             columns=risk_columns([("ID", "customer_id"), ("Region", "region"), ("Segmen", "segment"),
                                   ("Kontrak", "contract_type"), ("Tenure", "tenure_months"),
                                   ("Keluhan", "num_complaints"), ("Prob. churn", "churn_probability"),
                                   ("Risiko", "risk_category")]),
             style_data_conditional=RISK_COND, **TABLE_STYLE), marginTop="10px"),
])

# ============ 6. HALAMAN 3 — PREDIKSI SINGLE OBSERVASI ============
def num_input(id_, lab, val, mn, mx, step=1):
    return html.Div([label(lab), dcc.Input(id=id_, type="number", value=val, min=mn, max=mx,
                                           step=step, style={"width": "100%"})])
def dd_input(id_, lab, opts):
    return html.Div([label(lab), dcc.Dropdown(id=id_, options=opts, value=opts[0], clearable=False)])

page_single = html.Div(id="page-single", children=[
    page_header("Prediksi Single Observasi",
                "Simulasikan profil satu pelanggan untuk melihat probabilitas churn dan rekomendasi tindakan"),
    html.Div(style={**GRID2, "gridTemplateColumns": "minmax(0, 3fr) minmax(0, 2fr)"}, children=[
        card(html.H4("Profil pelanggan", style={"margin": "0 0 8px", "fontSize": "14px"}),
             html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))",
                             "gap": "10px"}, children=[
                 dd_input("p-region", "Region", CATS["region"]),
                 dd_input("p-segment", "Segmen", CATS["segment"]),
                 dd_input("p-contract", "Kontrak", CATS["contract_type"]),
                 dd_input("p-payment", "Pembayaran", CATS["payment_method"]),
                 dd_input("p-service", "Layanan", CATS["internet_service"]),
                 num_input("p-tenure", "Tenure (bulan)", 6, 1, 72),
                 num_input("p-monthly", "Tagihan bulanan (Rp)", 450000, 100000, 2000000, 10000),
                 num_input("p-complaints", "Jumlah keluhan", 2, 0, 10),
                 num_input("p-calls", "Panggilan CS", 3, 0, 15)]),
             html.Br(),
             html.Button("Jalankan prediksi", id="btn-predict", n_clicks=0, style=BTN)),
        card(html.H4("Hasil prediksi", style={"margin": "0 0 4px", "fontSize": "14px"}),
             html.Div(id="pred-output", children=html.P("Isi form lalu klik tombol.",
                                                        style={"margin": 0, "color": C["muted"]})),
             graph("g-gauge", 200, figure=empty_fig("Belum ada prediksi")))]),
])

# ============ 7. HALAMAN 4 — PREDIKSI MULTI OBSERVASI ============
page_batch = html.Div(id="page-batch", children=[
    page_header("Prediksi Multi Observasi",
                "Unggah file CSV berisi banyak pelanggan untuk diprediksi sekaligus"),
    card(html.Div(style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}, children=[
        html.Div(style={"flex": 2}, children=[
            dcc.Upload(id="upload-csv", accept=".csv", multiple=False,
                       children=html.Div(["Seret & lepas file CSV di sini, atau ",
                                          html.A("pilih file", style={"color": C["accent"],
                                                                      "fontWeight": 600})]),
                       style={"border": f"2px dashed {C['accent']}", "borderRadius": "10px",
                              "padding": "28px", "textAlign": "center", "background": "#F0F4FB",
                              "cursor": "pointer"}),
            html.Div(id="upload-status", style={"marginTop": "10px"})]),
        html.Div(style={"flex": 1, "fontSize": "13px"}, children=[
            html.B("Kolom wajib:"),
            html.Ul([html.Li(html.Code(c)) for c in FEATURES if c != "total_charges"],
                    style={"margin": "4px 0", "paddingLeft": "18px"}),
            html.Div(["Opsional: ", html.Code("customer_id"), ", ", html.Code("total_charges"),
                      " (jika kosong dihitung dari tagihan × tenure)."], style={"color": C["muted"]}),
            html.Br(),
            html.Button("⬇ Unduh template CSV", id="btn-template", n_clicks=0, style=BTN_OUTLINE),
            dcc.Download(id="dl-template")])])),
    html.Div(id="batch-result", style={"display": "none", "marginTop": "10px"}, children=[
        html.Div(id="batch-kpi", style=KPI_ROW),
        html.Div(style={**GRID2, "gridTemplateColumns": "minmax(0, 2fr) minmax(0, 3fr)"},
                 children=[card(graph("g-batch-donut", 280)), card(graph("g-batch-hist", 280))]),
        card(html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                      children=[html.H4("Hasil prediksi per pelanggan", style={"margin": 0}),
                                html.Button("⬇ Unduh hasil (CSV)", id="btn-dl-batch", n_clicks=0, style=BTN),
                                dcc.Download(id="dl-batch")]),
             html.Br(),
             dash_table.DataTable(id="tbl-batch", page_size=15, sort_action="native",
                                  filter_action="native", style_data_conditional=RISK_COND, **TABLE_STYLE),
             marginTop="12px")]),
    dcc.Store(id="store-batch"),
])

# ============ 8. HALAMAN 5 — TIM DEVELOPER ============
def initials(nama): return "".join(w[0] for w in nama.split()[:2]).upper()
def team_card(m):
    return card(
        html.Div(initials(m["nama"]), style={"width": "64px", "height": "64px", "borderRadius": "50%",
                                             "background": C["accent"], "color": "white", "display": "flex",
                                             "alignItems": "center", "justifyContent": "center",
                                             "fontSize": "22px", "fontWeight": 700, "margin": "0 auto 10px"}),
        html.Div(m["nama"], style={"fontWeight": 700, "fontSize": "16px"}),
        html.Div(m["peran"], style={"color": C["accent"], "fontSize": "13px", "fontWeight": 600,
                                    "margin": "4px 0 8px"}),
        html.Div(m["tugas"], style={"color": C["muted"], "fontSize": "13px"}),
        html.Div(m["email"], style={"fontSize": "12px", "marginTop": "8px"}),
        textAlign="center")

page_team = html.Div(id="page-team", children=[
    page_header("Tim Developer", "Tim yang membangun model dan dashboard ini"),
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(220px, 1fr))",
                    "gap": "12px"}, children=[team_card(m) for m in TEAM]),
    card(html.H4("Tentang proyek", style={"marginTop": 0}),
         html.Ul([
             html.Li(f"Model: {META['model_name']} v{META['model_version']} "
                     f"(threshold keputusan {META['decision_threshold']})"),
             html.Li(f"Data latihan: {len(DF):,} pelanggan, pendaftaran "
                     f"{DF.signup_date.min():%b %Y} – {DF.signup_date.max():%b %Y}"),
             html.Li("Teknologi: Python, pandas, scikit-learn, Plotly, Dash")]),
         marginTop="12px"),
])

# ============ 9. APP + LAYOUT ============
app = Dash(__name__, title="Final Dashboard — Customer Churn")
server = app.server                                  # untuk deployment (gunicorn)

# CSS global: aplikasi dikunci setinggi & selebar layar; hanya area konten yang di-scroll vertikal
CSS = f"""
html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
*, *::before, *::after {{ box-sizing: border-box; }}
.sidebar {{ width: 230px; flex-shrink: 0; height: 100vh; overflow-y: auto; overflow-x: hidden;
           background: {C['side']}; color: white; padding: 14px 12px; transition: width .2s ease; }}
.sidebar.collapsed {{ width: 60px; padding: 14px 8px; }}
.sidebar.collapsed .side-hide {{ display: none !important; }}
.sidebar.collapsed .toggle-btn {{ margin: 0 auto; }}
.nav-icon {{ width: 22px; text-align: center; flex-shrink: 0; font-size: 15px; }}
.toggle-btn {{ background: transparent; color: white; border: 1px solid #3A4766; border-radius: 6px;
              width: 32px; height: 32px; cursor: pointer; font-size: 16px; flex-shrink: 0; }}
.toggle-btn:hover {{ background: #2C3A5C; }}
/* Tombol pop up chart: muncul saat kursor di atas chart */
.expand-btn {{ position: absolute; top: 2px; right: 2px; z-index: 5; width: 28px; height: 28px;
              border: 1px solid #D5DCE8; border-radius: 6px; background: white; color: {C['accent']};
              font-size: 16px; line-height: 1; cursor: pointer; opacity: 0; transition: opacity .15s; }}
.chart-wrap:hover .expand-btn {{ opacity: 1; }}
.expand-btn:hover {{ background: {C['accent']}; color: white; }}
.chart-wrap .modebar-container {{ right: 34px !important; }}
/* Pop up */
.modal-overlay {{ position: fixed; inset: 0; background: rgba(15, 23, 42, .55); z-index: 1000;
                 display: flex; align-items: center; justify-content: center; }}
.modal-box {{ background: white; border-radius: 12px; width: 92vw; height: 88vh; padding: 12px 16px;
             display: flex; flex-direction: column; box-shadow: 0 20px 50px rgba(0,0,0,.3); }}
.modal-close {{ border: 0; background: #EEF1F6; border-radius: 6px; width: 32px; height: 32px;
               font-size: 16px; cursor: pointer; }}
.modal-close:hover {{ background: #C44E52; color: white; }}
/* Checklist di sidebar: Dash 4 memberi latar putih + teks gelap pada tiap opsi -> paksa transparan & putih */
.side-check .dash-options-list-option, .side-check label {{ background: transparent !important;
    color: white !important; font-size: 12px; padding: 2px 0; }}
.side-check .dash-options-list-option-text, .side-check span {{ color: white !important; }}
.side-check .dash-options-list-option.dash-checklist-inline {{ margin-right: 10px; }}
.side-check input {{ margin-right: 5px; accent-color: #6E9BE8; }}
.side-slider {{ padding: 0 4px; }}
.side-slider .rc-slider-mark-text, .side-slider [class*="mark"] {{ color: white !important; font-size: 11px; }}
.sidebar .Select-control, .sidebar .dash-dropdown {{ font-size: 12px; }}
"""
app.index_string = f"""<!DOCTYPE html>
<html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}<style>{CSS}</style></head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

app.layout = html.Div(style={"display": "flex", "height": "100vh", "width": "100%", "overflow": "hidden",
                             "background": C["bg"], "fontFamily": "Inter, Arial, sans-serif",
                             "color": C["ink"]}, children=[
    dcc.Location(id="url"),
    sidebar,
    html.Div(style={"flex": 1, "minWidth": 0, "height": "100vh", "overflowY": "auto", "overflowX": "hidden",
                    "padding": "14px 16px"},
             children=[page_overview, page_eda, page_single, page_batch, page_team]),
    # Pop up chart (tersembunyi sampai tombol ⤢ diklik)
    html.Div(id="modal", className="modal-overlay", style={"display": "none"}, children=[
        html.Div(className="modal-box", children=[
            html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                     children=[html.Div(id="modal-title", style={"fontWeight": 700, "fontSize": "16px"}),
                               html.Button("✕", id="modal-close", n_clicks=0, className="modal-close",
                                           title="Tutup")]),
            dcc.Graph(id="modal-graph", style={"flex": 1, "minHeight": 0},
                      config={"displaylogo": False, "responsive": True})])]),
])

# ============ 10. CALLBACKS ============
# --- Navigasi: tampilkan halaman aktif, sorot menu, tampilkan filter hanya di Overview & EDA
@callback([Output(f"page-{k}", "style") for k, *_ in PAGES] +
          [Output(f"nav-{k}", "style") for k, *_ in PAGES] +
          [Output("filter-panel", "style")],
          Input("url", "pathname"))
def navigate(pathname):
    aktif = PATH_TO_PAGE.get(pathname, "overview")
    pages = [{"display": "block"} if k == aktif else {"display": "none"} for k, *_ in PAGES]
    navs = [NAV_ACTIVE if k == aktif else NAV_BASE for k, *_ in PAGES]
    return pages + navs + [{"display": "block" if aktif in ("overview", "eda") else "none"}]


# --- Pop up chart: salin figure chart yang diklik ke dalam modal berukuran besar
@callback(Output("modal", "style"), Output("modal-graph", "figure"), Output("modal-title", "children"),
          Input({"type": "expand", "index": ALL}, "n_clicks"), Input("modal-close", "n_clicks"),
          *[State(g, "figure") for g in GRAPH_IDS], prevent_initial_call=True)
def toggle_modal(_expand, _close, *figs):
    trig = ctx.triggered_id
    if trig == "modal-close":
        return {"display": "none"}, go.Figure(), ""
    if not isinstance(trig, dict) or not ctx.triggered[0]["value"]:
        raise PreventUpdate                            # abaikan render awal tombol
    fig = figs[GRAPH_IDS.index(trig["index"])]
    if not fig:
        raise PreventUpdate
    fig = go.Figure(fig)
    judul = fig.layout.title.text or "Detail chart"
    fig.update_layout(title_text="", height=None, autosize=True, font_size=13,
                      margin=dict(l=20, r=20, t=30, b=20))
    return {"display": "flex"}, fig, judul


# --- Perkecil / perbesar sidebar
@callback(Output("sidebar", "className"), Input("btn-toggle", "n_clicks"))
def toggle_sidebar(n):
    return "sidebar collapsed" if (n or 0) % 2 else "sidebar"


def apply_filters(region, segment, kontrak, risk, tenure):
    d = DF
    if region: d = d[d.region.isin(region)]
    if segment: d = d[d.segment.isin(segment)]
    d = d[d.contract_type.isin(kontrak or []) & d.risk_category.isin(risk or [])]
    return d[d.tenure_months.between(*tenure)]

FILTERS = [Input("f-region", "value"), Input("f-segment", "value"), Input("f-kontrak", "value"),
           Input("f-risk", "value"), Input("f-tenure", "value")]


# --- Halaman 1: Data Overview
@callback(Output("kpi-row", "children"), Output("g-status", "figure"), Output("dataset-info", "children"),
          Output("tbl-profile", "data"), Output("tbl-profile", "columns"),
          Output("tbl-describe", "data"), Output("tbl-describe", "columns"),
          Output("tbl-preview", "data"), Output("tbl-preview", "columns"), *FILTERS)
def update_overview(region, segment, kontrak, risk, tenure):
    d = apply_filters(region, segment, kontrak, risk, tenure)
    raw_cols = [c for c in raw.columns]
    profile = pd.DataFrame({
        "Kolom": raw_cols,
        "Tipe": [str(DF[c].dtype) for c in raw_cols],
        "Missing (mentah)": [int(RAW_MISSING[c]) for c in raw_cols],
        "Nilai unik": [int(DF[c].nunique()) for c in raw_cols],
        "Contoh": [str(DF[c].iloc[0]) for c in raw_cols]})
    prof_cols = [{"name": c, "id": c} for c in profile.columns]
    preview_ids = ["customer_id", "signup_date"] + FEATURES + ["status"]
    prev_cols = [{"name": COL_LABEL.get(c, c), "id": c} for c in preview_ids]
    if d.empty:
        return ([kpi("Pelanggan", "0", "sesuai filter")], empty_fig(), html.P("Tidak ada data."),
                profile.to_dict("records"), prof_cols, [], [], [], prev_cols)

    churn_n = int(d.churn.sum())
    kpis = [kpi("Pelanggan", f"{len(d):,}", f"dari {len(DF):,} total"),
            kpi("Churn aktual", f"{churn_n:,}", f"churn rate {d.churn.mean():.1%}"),
            kpi("Rata-rata tenure", f"{d.tenure_months.mean():.1f} bln",
                f"median {d.tenure_months.median():.0f} bln"),
            kpi("Pendapatan bulanan", rupiah(d.monthly_charges.sum()),
                f"rata-rata {rupiah(d.monthly_charges.mean())}/pelanggan")]

    g = px.pie(d, names="status", hole=0.55, color="status", color_discrete_map=STATUS_COLOR,
               category_orders=CAT_ORDERS, title="Komposisi status churn aktual")
    g.update_traces(textinfo="percent+label", sort=False)
    g.add_annotation(text=f"<b>{len(d):,}</b><br>pelanggan", showarrow=False, font_size=14)
    g.update_layout(showlegend=False)

    info_rows = [("Jumlah baris (sesuai filter)", f"{len(d):,}"),
                 ("Jumlah kolom", f"{raw.shape[1]}"),
                 ("Variabel numerik", ", ".join(NUM_COLS)),
                 ("Variabel kategorik", ", ".join(CAT_COLS)),
                 ("Variabel target", "churn (0 = Bertahan, 1 = Churn)"),
                 ("Periode pendaftaran", f"{d.signup_date.min():%d %b %Y} – {d.signup_date.max():%d %b %Y}"),
                 ("Total missing value (mentah)", f"{int(RAW_MISSING.sum()):,} sel"),
                 ("Baris duplikat", f"{int(raw.duplicated().sum()):,}")]
    info = html.Table(style={"width": "100%", "fontSize": "13px", "borderCollapse": "collapse"}, children=[
        html.Tr([html.Td(k, style={"color": C["muted"], "padding": "6px 8px 6px 0",
                                   "borderBottom": "1px solid #EEF1F6", "width": "45%"}),
                 html.Td(v, style={"fontWeight": 600, "padding": "6px 0",
                                   "borderBottom": "1px solid #EEF1F6"})]) for k, v in info_rows])

    desc = d[NUM_COLS].describe().T.round(2).reset_index().rename(columns={"index": "variabel"})
    desc["variabel"] = desc["variabel"].map(COL_LABEL)
    desc_cols = [{"name": c, "id": c} for c in desc.columns]

    prev = d[preview_ids].copy()
    prev["signup_date"] = prev["signup_date"].dt.strftime("%Y-%m-%d")
    return (kpis, tidy(g), info, profile.to_dict("records"), prof_cols,
            desc.to_dict("records"), desc_cols, prev.head(500).to_dict("records"), prev_cols)


# --- Halaman 2: EDA (univariat/bivariat)
@callback(Output("g-num-dist", "figure"), Output("g-cat-rate", "figure"), Output("g-box", "figure"),
          Output("g-corr", "figure"), Input("eda-num", "value"), Input("eda-cat", "value"), *FILTERS)
def update_eda_explore(num, cat, region, segment, kontrak, risk, tenure):
    d = apply_filters(region, segment, kontrak, risk, tenure)
    if d.empty:
        f = empty_fig()
        return f, f, f, f
    g1 = px.histogram(d, x=num, color="status", barmode="overlay", nbins=30, opacity=0.7,
                      marginal="box", color_discrete_map=STATUS_COLOR, category_orders=CAT_ORDERS,
                      template="plotly_white", title=f"Distribusi {COL_LABEL[num]} per status churn",
                      labels={num: COL_LABEL[num], "status": ""})
    cr = d.groupby(cat).agg(churn_rate=("churn", "mean"), n=("customer_id", "count")).reset_index()
    cr = cr.sort_values("churn_rate", ascending=False)
    g2 = px.bar(cr, x=cat, y="churn_rate", text_auto=".1%", template="plotly_white",
                color_discrete_sequence=[STATUS_COLOR["Churn"]], hover_data={"n": True},
                title=f"Churn rate aktual per {COL_LABEL[cat].lower()}",
                labels={cat: "", "churn_rate": "Churn rate", "n": "Pelanggan"})
    g2.add_hline(y=d.churn.mean(), line_dash="dash", annotation_text="rata-rata")
    g2.update_yaxes(tickformat=".0%")
    g3 = px.box(d, x=cat, y=num, color="status", color_discrete_map=STATUS_COLOR,
                category_orders=CAT_ORDERS, template="plotly_white",
                title=f"{COL_LABEL[num]} per {COL_LABEL[cat].lower()} dan status",
                labels={cat: "", num: COL_LABEL[num], "status": ""})
    corr = d[NUM_COLS + ["churn", "churn_probability"]].corr().round(2)
    SINGKAT = {"tenure_months": "Tenure", "monthly_charges": "Tagihan", "total_charges": "Total",
               "num_complaints": "Keluhan", "support_calls": "Panggilan", "churn": "Churn",
               "churn_probability": "Prob."}
    nama = [SINGKAT[c] for c in corr.columns]
    g4 = px.imshow(corr.values, x=nama, y=nama, text_auto=True, zmin=-1, zmax=1,
                   color_continuous_scale="RdBu_r", title="Korelasi variabel numerik", aspect="auto")
    g4.update_layout(template="plotly_white")
    return tidy(g1), tidy(g2), tidy(g3), tidy(g4, legend_top=False)


# --- Halaman 2: EDA (risiko hasil prediksi)
@callback(Output("g-donut", "figure"), Output("g-region-risk", "figure"),
          Output("g-trend", "figure"), Output("g-contract", "figure"),
          Output("g-risk", "figure"), Output("g-actual-pred", "figure"),
          Output("tbl-risk", "data"), Output("tbl-title", "children"), *FILTERS)
def update_eda_risk(region, segment, kontrak, risk, tenure):
    d = apply_filters(region, segment, kontrak, risk, tenure)
    if d.empty:
        f = empty_fig()
        return f, f, f, f, f, f, [], "Tidak ada data"
    g0 = px.pie(d, names="risk_category", hole=0.55, color="risk_category",
                color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                title="Komposisi kategori risiko")
    g0.update_traces(textinfo="percent+label", sort=False)
    tinggi = (d.risk_category == "Tinggi").sum()
    g0.add_annotation(text=f"<b>{tinggi:,}</b><br>risiko tinggi", showarrow=False, font_size=14)
    g0.update_layout(showlegend=False)
    rr = (d.assign(tinggi=d.risk_category.eq("Tinggi")).groupby("region", as_index=False)["tinggi"]
            .mean().sort_values("tinggi"))
    g5 = px.bar(rr, x="tinggi", y="region", orientation="h", text_auto=".1%",
                template="plotly_white", color_discrete_sequence=[RISK_COLOR["Tinggi"]],
                title="Porsi risiko tinggi per region",
                labels={"tinggi": "Porsi risiko tinggi", "region": ""})
    g5.update_xaxes(tickformat=".0%")
    tr = d.groupby("periode").agg(aktual=("churn", "mean"), prediksi=("churn_probability", "mean")).reset_index()
    g1 = px.line(tr.melt("periode", var_name="seri", value_name="rate"), x="periode", y="rate",
                 color="seri", markers=True, template="plotly_white",
                 title="Churn aktual vs prediksi per kuartal",
                 labels={"periode": "Kuartal pendaftaran", "rate": "Rate", "seri": ""})
    g1.update_yaxes(tickformat=".0%")
    ck = d.groupby(["contract_type", "risk_category"]).size().reset_index(name="n")
    g2 = px.bar(ck, x="contract_type", y="n", color="risk_category", color_discrete_map=RISK_COLOR,
                category_orders=CAT_ORDERS, template="plotly_white", title="Komposisi risiko per jenis kontrak",
                labels={"contract_type": "", "n": "Pelanggan", "risk_category": "Risiko"})
    g3 = px.histogram(d, x="churn_probability", nbins=30, color="risk_category",
                      color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                      template="plotly_white", title="Distribusi probabilitas churn",
                      labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"})
    g3.add_vline(x=META["decision_threshold"], line_dash="dash", annotation_text="threshold")
    rg = d.groupby("region").agg(aktual=("churn", "mean"), prediksi=("churn_probability", "mean"),
                                 n=("customer_id", "count")).reset_index()
    g4 = px.scatter(rg, x="aktual", y="prediksi", size="n", text="region", template="plotly_white",
                    title="Region: aktual vs prediksi (ukuran = n)",
                    labels={"aktual": "Churn rate aktual", "prediksi": "Rata-rata prediksi"})
    g4.update_traces(textposition="top center", cliponaxis=False)
    pad = 0.03
    g4.update_xaxes(tickformat=".0%", range=[rg.aktual.min() - pad, rg.aktual.max() + pad])
    g4.update_yaxes(tickformat=".0%", range=[rg.prediksi.min() - pad, rg.prediksi.max() + pad])
    top = d.sort_values("churn_probability", ascending=False).head(200)
    return (*[tidy(f) for f in (g0, g5, g1, g2, g3, g4)], top.to_dict("records"),
            f"Prioritas retensi — {min(len(d), 200)} pelanggan dengan probabilitas tertinggi")


# --- Halaman 3: Prediksi single observasi
SARAN = {"Tinggi": "Prioritaskan retensi: hubungi dalam 48 jam, tawarkan kontrak tahunan.",
         "Sedang": "Pantau: kirim survei kepuasan dan penawaran loyalitas.",
         "Rendah": "Pertahankan layanan standar."}

@callback(Output("pred-output", "children"), Output("g-gauge", "figure"),
          Input("btn-predict", "n_clicks"),
          State("p-region", "value"), State("p-segment", "value"), State("p-contract", "value"),
          State("p-payment", "value"), State("p-service", "value"), State("p-tenure", "value"),
          State("p-monthly", "value"), State("p-complaints", "value"), State("p-calls", "value"),
          prevent_initial_call=True)
def run_prediction(n, region, segment, contract, payment, service, tenure, monthly, complaints, calls):
    if None in (tenure, monthly, complaints, calls):
        raise PreventUpdate
    row = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": monthly,
                         "total_charges": monthly * tenure, "num_complaints": complaints,
                         "support_calls": calls, "region": region, "segment": segment,
                         "contract_type": contract, "payment_method": payment,
                         "internet_service": service}])
    errs = validate_input(row, META)
    if errs:
        return html.Div([html.P(e, style={"color": "crimson"}) for e in errs]), empty_fig("Input tidak valid")
    r = predict(MODEL, row, META).iloc[0]
    out = html.Div([
        html.Div(f"{r.churn_probability:.1%}", style={"fontSize": "32px", "fontWeight": 700,
                                                     "color": RISK_COLOR[r.risk_category]}),
        html.Div(f"Prediksi: {'Churn' if r.predicted_churn else 'Bertahan'} · Risiko {r.risk_category}"),
        html.P(SARAN[r.risk_category], style={"color": C["muted"]})])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=r.churn_probability * 100,
                                   number={"suffix": "%"}, gauge={
                                       "axis": {"range": [0, 100]},
                                       "bar": {"color": RISK_COLOR[r.risk_category]},
                                       "steps": [{"range": [0, 30], "color": "#E6F4EA"},
                                                 {"range": [30, 60], "color": "#FDF0E6"},
                                                 {"range": [60, 100], "color": "#FBE3E3"}]}))
    gauge.update_layout(margin=dict(t=10, b=0, l=25, r=25))
    return out, gauge


# --- Halaman 4: Prediksi multi observasi (upload CSV)
@callback(Output("dl-template", "data"), Input("btn-template", "n_clicks"), prevent_initial_call=True)
def download_template(n):
    contoh = DF[["customer_id"] + FEATURES].sample(5, random_state=42)
    return dcc.send_data_frame(contoh.to_csv, "template_prediksi_churn.csv", index=False)


def parse_csv(contents):
    _, isi = contents.split(",", 1)
    teks = base64.b64decode(isi).decode("utf-8-sig")
    return pd.read_csv(io.StringIO(teks), sep=None, engine="python")   # deteksi otomatis , atau ;


def status_box(pesan, warna):
    return html.Div(pesan, style={"padding": "10px 12px", "borderRadius": "8px", "fontSize": "13px",
                                  "background": {"ok": "#E6F4EA", "err": "#FBE3E3"}[warna],
                                  "color": {"ok": "#1E6B34", "err": "#8B1E24"}[warna]})


@callback(Output("upload-status", "children"), Output("batch-result", "style"),
          Output("batch-kpi", "children"), Output("g-batch-donut", "figure"), Output("g-batch-hist", "figure"),
          Output("tbl-batch", "data"), Output("tbl-batch", "columns"), Output("store-batch", "data"),
          Input("upload-csv", "contents"), State("upload-csv", "filename"), prevent_initial_call=True)
def run_batch(contents, filename):
    hidden = {"display": "none"}
    gagal = lambda msg: (status_box(msg, "err"), hidden, [], {}, {}, [], [], None)
    if contents is None:
        raise PreventUpdate
    if not filename.lower().endswith(".csv"):
        return gagal(f"File '{filename}' bukan CSV.")
    try:
        df = parse_csv(contents)
    except Exception as e:
        return gagal(f"Gagal membaca file: {e}")
    df.columns = df.columns.str.strip()
    wajib = [c for c in FEATURES if c != "total_charges"]
    hilang = [c for c in wajib if c not in df.columns]
    if hilang:
        return gagal(f"Kolom wajib tidak ditemukan: {', '.join(hilang)}")
    if df.empty:
        return gagal("File tidak berisi data.")

    df = df.copy()
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "total_charges" not in df.columns:
        df["total_charges"] = pd.NA
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"])
    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", [f"ROW-{i + 1}" for i in range(len(df))])
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip()

    # Baris dengan nilai kosong di kolom wajib dilewati agar tidak menggagalkan seluruh batch
    lengkap = df[wajib].notna().all(axis=1)
    dilewati = int((~lengkap).sum())
    df = df[lengkap]
    if df.empty:
        return gagal("Semua baris memiliki nilai kosong pada kolom wajib.")

    errs = validate_input(df[FEATURES], META)
    if errs:
        return gagal(html.Div([html.B("Validasi gagal:"), html.Ul([html.Li(e) for e in errs])]))
    try:
        hasil = predict(MODEL, df, META)
    except Exception as e:
        return gagal(f"Prediksi gagal: {e}")
    hasil = hasil.sort_values("churn_probability", ascending=False)
    hasil["predicted_churn"] = hasil["predicted_churn"].map({0: "Bertahan", 1: "Churn"})
    hasil["rekomendasi"] = hasil["risk_category"].map(SARAN)

    n = len(hasil)
    tinggi = hasil.risk_category.eq("Tinggi")
    kpis = [kpi("Pelanggan diprediksi", f"{n:,}", f"{dilewati} baris dilewati" if dilewati else "semua baris valid"),
            kpi("Prediksi churn", f"{hasil.predicted_churn.eq('Churn').sum():,}",
                f"{hasil.predicted_churn.eq('Churn').mean():.1%} dari total"),
            kpi("Rata-rata prob. churn", f"{hasil.churn_probability.mean():.1%}", "prediksi model"),
            kpi("Risiko tinggi", f"{tinggi.sum():,}",
                rupiah(hasil.loc[tinggi, "monthly_charges"].sum()) + "/bln berisiko")]
    g1 = px.pie(hasil, names="risk_category", hole=0.55, color="risk_category",
                color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                title="Komposisi kategori risiko")
    g1.update_traces(textinfo="percent+label+value", sort=False)
    g1.update_layout(showlegend=False)
    g2 = px.histogram(hasil, x="churn_probability", nbins=20, color="risk_category",
                      color_discrete_map=RISK_COLOR, category_orders={"risk_category": RISK_ORDER},
                      template="plotly_white", title="Distribusi probabilitas churn",
                      labels={"churn_probability": "Probabilitas churn", "risk_category": "Risiko"})
    g2.add_vline(x=META["decision_threshold"], line_dash="dash", annotation_text="threshold")

    tampil = ["customer_id"] + FEATURES + ["churn_probability", "predicted_churn", "risk_category", "rekomendasi"]
    kolom = risk_columns([(COL_LABEL.get(c, {"customer_id": "ID", "churn_probability": "Prob. churn",
                                              "predicted_churn": "Prediksi", "risk_category": "Risiko",
                                              "rekomendasi": "Rekomendasi"}.get(c, c)), c) for c in tampil])
    data = hasil[tampil].to_dict("records")
    msg = f"✔ '{filename}' berhasil diproses: {n:,} pelanggan diprediksi."
    if dilewati:
        msg += f" {dilewati} baris dilewati karena ada nilai kosong."
    return status_box(msg, "ok"), {"display": "block", "marginTop": "12px"}, kpis, tidy(g1), tidy(g2), data, kolom, data


@callback(Output("dl-batch", "data"), Input("btn-dl-batch", "n_clicks"), State("store-batch", "data"),
          prevent_initial_call=True)
def download_batch(n, data):
    if not data:
        raise PreventUpdate
    return dcc.send_data_frame(pd.DataFrame(data).to_csv, "hasil_prediksi_churn.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, port=8053)
