from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from utils import utils
import datetime
import pytz
import plotly.graph_objs as go


tz = "America/Santiago"

app = Dash(__name__)

since = int(
    (
        datetime.datetime.now(pytz.timezone(tz)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        - datetime.timedelta(days=30)
    ).timestamp()
    * 1000
)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Tráfico vehicular\nAntofagasta"),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="dd_type",
                            options=[
                                {"label": "Todos", "value": "all"},
                                {"label": "Accidentes", "value": "ACCIDENT"},
                                {"label": "Congestión", "value": "JAM"},
                                {"label": "Peligros", "value": "HAZARD"},
                                {"label": "Caminos cerrados", "value": "ROAD_CLOSED"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                        dcc.DatePickerRange(
                            start_date_placeholder_text="Desde",
                            end_date_placeholder_text="Hasta",
                            id="date_range",
                            min_date_allowed=datetime.datetime.fromtimestamp(
                                since / 1000, pytz.timezone(tz)
                            ),
                            max_date_allowed=datetime.datetime.now(pytz.timezone(tz)),
                            initial_visible_month=datetime.datetime.fromtimestamp(
                                since / 1000, pytz.timezone(tz)
                            ),
                            display_format="DD-MM-YYYY",
                            calendar_orientation="vertical",
                            start_date=datetime.datetime.fromtimestamp(
                                since / 1000, pytz.timezone(tz)
                            ),
                            end_date=datetime.datetime.now(pytz.timezone(tz)),
                            first_day_of_week=1,
                        ),
                    ],
                    className="controls",
                ),
            ],
            className="header",
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="map",
                            className="plot",
                        ),
                    ],
                    className="plot-container",
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="hourly",
                        ),
                    ],
                    className="plot-container",
                ),
            ],
            className="first-row",
        ),
    ],
    className="container",
)


@app.callback(
    [Output("map", "figure"), Output("hourly", "figure")],
    [
        Input("dd_type", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
    ],
)
def update_graphs(kind, start_date, end_date):
    if start_date is None or end_date is None:
        return go.Figure(), go.Figure()

    start = int((datetime.datetime.fromisoformat(start_date)).timestamp() * 1000)
    end = int((datetime.datetime.fromisoformat(end_date)).timestamp() * 1000)

    alerts = utils.load_data("alerts", mode="between", between=(start, end))
    alerts = alerts.to_gdf(tz=tz)
    alerts = utils.freq_nearby(alerts, nearby_meters=200)
    alerts["freq"] = alerts.apply(lambda x: x["freq"] if x["freq"] > 0 else 1, axis=1)
    alerts = utils.filter_nearby(alerts, threshold=200)

    if kind == "all":
        events = utils.extract_event(
            alerts, ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]
        )
    else:
        events = utils.extract_event(alerts, [kind])
    hourly = utils.hourly_group(events).reset_index()

    names = {
        "all": "Eventos",
        "ACCIDENT": "Accidentes",
        "JAM": "Congestión",
        "HAZARD": "Peligros",
        "ROAD_CLOSED": "Caminos cerrados",
    }

    if kind != "all":
        map_data = alerts[alerts["type"] == kind].copy()
    else:
        map_data = alerts

    map_fig = go.Figure(
        px.scatter_map(
            map_data,
            lat="y",
            lon="x",
            color="type",
            hover_name="uuid",
            size="freq",
            zoom=10.5,
            labels={
                "type": "Tipo de evento",
                "freq": "Frecuencia",
                "x": "Longitud",
                "y": "Latitud",
            },
            opacity=0.8,
        )
    )
    hourly_fig = go.Figure(
        go.Bar(
            x=hourly["hour"],
            y=hourly["s"],
            marker_color="lightskyblue",
            name="Accidentes",
            opacity=0.8,
        ),
    )

    hourly_fig.update_layout(
        paper_bgcolor="#333",
        plot_bgcolor="#333",
        title=dict(
            text=f"{names[kind].capitalize()} por hora del día",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=20,
            ),
            xanchor="left",
            yanchor="top",
            pad=dict(b=30, l=50),
        ),
        font=dict(color="#ccc"),
        grid_ygap=1,
    )

    map_fig.update_layout(
        paper_bgcolor="#333",
        plot_bgcolor="#333",
        hoverlabel_font=dict(color="#fff"),
        showlegend=False,
        title=dict(
            text=f"{names[kind]} de tráfico",
            font=dict(
                color="#ccc",
                weight=600,
                family="Verdana",
                size=20,
            ),
            xanchor="left",
            yanchor="top",
            pad=dict(b=30, l=50),
        ),
        margin=dict(t=80, b=20, l=40, r=30),
        map_style="dark",
    )

    if kind == "all":
        map_fig.update_layout(
            showlegend=True,
            legend_font=dict(color="#ccc"),
        )

    return map_fig, hourly_fig
