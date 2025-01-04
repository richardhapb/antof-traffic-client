import os
import datetime
import pytz
from dash import html, dcc
from dashboard.models import TimeRange

TZ = os.getenv("TZ", "America/Santiago")


def get_header(time_range: TimeRange):

    header = html.Div(
        [
            html.H1("Inteligencia de Tráfico para la Gestión Urbana en Antofagasta"),
            html.H2(
                "Análisis de patrones y eventos reportados por usuarios para gestión de tráfico",
                className="subtitle",
            ),
            html.Button("Instrucciones", "toggle_instructions"),
            html.Div(
                id="instructions",
                children=[
                    html.H3("Instrucciones:"),
                    html.Ul(
                        [
                            html.Li(
                                "El dashboard se divide en dos secciones principales, la primera corresponde a representaciones descriptivas de los datos, la cual se compone de los 6 primeros gráficos/tablas, siendo el de correlación el último. La segunda sección corresponde al modelo predictivo, compuesto por el mapa, las entradas de parámetros y la tabla de información."
                            ),
                            html.Li(
                                "La primera sección es controlada por los filtros de tipo de evento y rango, se exceptúa en el filtro de tipo el gráfico de correlación, para el caso de los últimos eventos solo se toma en cuenta el filtro de tipo."
                            ),
                            html.Li(
                                "Los gráficos son interactivos, se pueden filtrar por calle, haciendo click en la tabla de calles, como también en las leyendas"
                            ),
                            html.Li(
                                "El modelo predictivo está basado en un modelo de Machine Learning y segmentado por parámetros temporales (tipo de día, día de la semana, día del mes, hora del día) y parámetros geospaciales, los cuales se resumen en el segmento asociado."
                            ),
                            html.Li(
                                "La tabla del modelo predictivo también es interactiva, se puede visualizar el segmento asociado haciendo click en la fila correspondiente."
                            ),
                            html.Li(
                                "Los datos están disponibles a partir del 1 de octubre de 2024, por lo que se pueden aplicar filtros de fecha hasta ese punto."
                            ),
                            html.Li(
                                "El objetivo del modelo predictivo es valorar la probabilidad de que un evento ocurra bajo los parámetros previamente establecidos, tiene aplicaciones como la de visualizar la ruta con menor probabilidad de evento desde un segmento A a un segmento B, también para poder identificar patrones para la gestión del tráfico."
                            ),
                        ],
                    ),
                ],
                className="instructions",
            ),
            html.Div(
                children=[
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
                            time_range.init_time / 1000, pytz.timezone(TZ)
                        ),
                        max_date_allowed=datetime.datetime.now(pytz.timezone(TZ)),
                        initial_visible_month=datetime.datetime.fromtimestamp(
                            time_range.selected_time / 1000, pytz.timezone(TZ)
                        ),
                        display_format="DD-MM-YYYY",
                        calendar_orientation="vertical",
                        start_date=datetime.datetime.fromtimestamp(
                            time_range.selected_time / 1000, pytz.timezone(TZ)
                        ),
                        end_date=datetime.datetime.fromtimestamp(
                            time_range.end_time / 1000, pytz.timezone(TZ)
                        ),
                        first_day_of_week=1,
                    ),
                ],
                className="controls",
            ),
        ],
        className="header",
    )

    return header
