from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import utils

app = Dash(__name__)

alerts, jams = utils.load_data()
alerts = utils.separate_coords(alerts)
alerts = utils.freq_nearby(alerts)

app.layout = html.Div(
    [
        html.H1("Tráfico vehicular - Antofagasta"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="map",
                            figure=px.scatter_map(
                                alerts,
                                lat="y",
                                lon="x",
                                color="type",
                                hover_name="uuid",
                                size="freq",
                                zoom=10,
                                height=700,
                                width=600,
                                title="Eventos de tráfico desde agosto-2024",
                                labels={
                                    "type": "Eventos",
                                    "freq": "Frecuencia",
                                    "x": "Longitud",
                                    "y": "Latitud",
                                },
                                template="plotly_white",
                            ),
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                    },
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
