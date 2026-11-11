# Kunci Sesi 7 no.3 — klik titik scatter menampilkan detail pelanggan
from pathlib import Path
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="sc", figure=px.scatter(df, x="tenure_months", y="monthly_charges",
                                         custom_data=["customer_id"], template="plotly_white")),
    html.Div(id="detail", children="Klik salah satu titik."),
])

@callback(Output("detail", "children"), Input("sc", "clickData"))
def tampil_detail(click):
    if not click:
        raise PreventUpdate
    cid = click["points"][0]["customdata"][0]
    r = df.loc[df.customer_id == cid].iloc[0]
    return html.Ul([html.Li(f"ID: {r.customer_id}"), html.Li(f"Region: {r.region}"),
                    html.Li(f"Kontrak: {r.contract_type}"), html.Li(f"Keluhan: {r.num_complaints}"),
                    html.Li(f"Status: {'Churn' if r.churn else 'Bertahan'}")])

if __name__ == "__main__":
    app.run(debug=True, port=8060)
