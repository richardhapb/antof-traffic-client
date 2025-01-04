import datetime
from dash import html, dcc, dash_table
from dashboard.models import Model


def get_ml_params_component(model: Model):
    ml_params = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Parámetros para el modelo"),
                            html.H3("Tipo de evento"),
                            dcc.Dropdown(
                                id="dd_type_ml",
                                options=[
                                    {
                                        "label": "Accidentes",
                                        "value": "ACCIDENT",
                                    },
                                    {"label": "Congestión", "value": "JAM"},
                                    {
                                        "label": "Peligros",
                                        "value": "HAZARD",
                                    },
                                    {
                                        "label": "Caminos cerrados",
                                        "value": "ROAD_CLOSED",
                                    },
                                ],
                                value="ACCIDENT",
                                clearable=False,
                            ),
                            html.H3("Hora y fecha"),
                            dcc.Dropdown(
                                id="dd_hour_ml",
                                options=[
                                    {"label": "00:00 - 00:59", "value": 0},
                                    {"label": "01:00 - 01:59", "value": 1},
                                    {"label": "02:00 - 02:59", "value": 2},
                                    {"label": "03:00 - 03:59", "value": 3},
                                    {"label": "04:00 - 04:59", "value": 4},
                                    {"label": "05:00 - 05:59", "value": 5},
                                    {"label": "06:00 - 06:59", "value": 6},
                                    {"label": "07:00 - 07:59", "value": 7},
                                    {"label": "08:00 - 08:59", "value": 8},
                                    {"label": "09:00 - 09:59", "value": 9},
                                    {"label": "10:00 - 10:59", "value": 10},
                                    {"label": "11:00 - 11:59", "value": 11},
                                    {"label": "12:00 - 12:59", "value": 12},
                                    {"label": "13:00 - 13:59", "value": 13},
                                    {"label": "14:00 - 14:59", "value": 14},
                                    {"label": "15:00 - 15:59", "value": 15},
                                    {"label": "16:00 - 16:59", "value": 16},
                                    {"label": "17:00 - 17:59", "value": 17},
                                    {"label": "18:00 - 18:59", "value": 18},
                                    {"label": "19:00 - 19:59", "value": 19},
                                    {"label": "20:00 - 20:59", "value": 20},
                                    {"label": "21:00 - 21:59", "value": 21},
                                    {"label": "22:00 - 22:59", "value": 22},
                                    {"label": "23:00 - 23:59", "value": 23},
                                ],
                                value=7,
                                clearable=False,
                            ),
                            dcc.DatePickerSingle(
                                id="date_ml",
                                min_date_allowed=datetime.date(2024, 10, 1),
                                max_date_allowed=datetime.date(2025, 12, 31),
                                initial_visible_month=datetime.date.today(),
                                date=datetime.date.today(),
                                display_format="DD/MM/YYYY",
                                first_day_of_week=1,
                            ),
                        ],
                        className="parameters_ml",
                    ),
                    html.Div(
                        [
                            html.H3("Segmentos y probabilidades"),
                            html.Button("Limpiar selección", id="table_ml_clear"),
                            dash_table.DataTable(
                                id="table_ml",
                                columns=[
                                    {"name": i, "id": i}
                                    for i in ["Segmento", "Probabilidad"]
                                ],
                                data=[],
                                filter_action="native",
                                sort_action="native",
                                page_action="native",
                                page_current=0,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "center",
                                    "height": "auto",
                                    "whiteSpace": "normal",
                                },
                                style_header={
                                    "backgroundColor": "rgba(30,30,30,0.6)",
                                    "color": "#ccc",
                                    "fontWeight": "semibold",
                                    "fontFamily": "Verdana",
                                    "fontSize": "1rem",
                                    "textAlign": "center",
                                    "border": "1px solid #555",
                                },
                                style_data={
                                    "backgroundColor": "rgba(30,30,30,0.6)",
                                    "color": "#fff",
                                    "fontWeight": "lighter",
                                    "fontFamily": "Verdana",
                                    "fontSize": "1rem",
                                    "textAlign": "center",
                                    "border": "1px solid #555",
                                },
                                style_filter={
                                    "backgroundColor": "rgba(30,30,30,0.6)",
                                    "color": "#000",
                                    "fontWeight": "semibold",
                                    "fontFamily": "Verdana",
                                    "fontSize": "1rem",
                                    "textAlign": "center",
                                    "border": "1px solid #555",
                                },
                            ),
                            html.P(
                                f"Versión del modelo: {model.last_model}",
                                id="model_version",
                            ),
                        ],
                        className="plot-container",
                    ),
                ],
                className="ml-zone",
            ),
        ],
        className="plot-container",
    )

    return ml_params
