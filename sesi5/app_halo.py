# cd dashboard_course 
#  Jalankan dari root proyek: .venv/bin/python sesi5/app_halo.py
# http://127.0.0.1:8050

from dash import Dash, Input, Output, dcc, html

app = Dash(__name__, title="Halo, Dash!")

app.layout = html.Div([
    html.H1("Halo, Dash! 👋"),
    html.P("Ini aplikasi data interaktif pertama saya."),
    html.Label("Siapa nama Anda?", htmlFor="nama-input"),
    dcc.Input(
        id="nama-input",
        type="text",
        placeholder="Ketik nama Anda",
    ),
    html.Div(id="pesan-sambutan"),
])


@app.callback(Output("pesan-sambutan", "children"), Input("nama-input", "value"))
def tampilkan_sambutan(nama):
    if not nama:
        return None

    return html.P(f"Selamat datang di pelatihan dashboard, {nama}!")


if __name__ == "__main__":
    app.run(debug=False, dev_tools_ui=False, port=8050)
