from dash import html, dash_table

street_table = html.Div(
    [
        html.H3("Eventos por calle"),
        dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in ["Calle", "Eventos"]],
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
            style_data_conditional=[
                {
                    "if": {"column_id": "Eventos"},
                    "width": "27%",
                    "textAlign": "center",
                },
                {
                    "if": {"column_id": "Calle"},
                    "width": "73%",
                    "textAlign": "left",
                },
            ],
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
        html.Button("Limpiar selección", id="table_clear"),
    ],
    className="plot-container",
)

last_events_table = html.Div(
    [
        html.H3("Últimos eventos reportados"),
        dash_table.DataTable(
            id="table_last",
            columns=[
                {"name": i, "id": i}
                for i in ["Tipo", "Fecha", "Hora", "Calle", "Segmento"]
            ],
            data=[],
            page_action="native",
            page_current=0,
            page_size=10,
            cell_selectable=False,
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center",
                "whiteSpace": "normal",
                "height": "auto",
                "wordBreak": "break-all",
            },
            style_data_conditional=[
                {"if": {"column_id": "Tipo"}, "width": "20%"},
                {"if": {"column_id": "Fecha"}, "width": "20%"},
                {"if": {"column_id": "Hora"}, "width": "10%"},
                {"if": {"column_id": "Calle"}, "width": "40%"},
                {"if": {"column_id": "Segmento"}, "width": "10%"},
            ],
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
        ),
    ],
    className="plot-container",
)
