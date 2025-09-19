from dash import html, dcc

hourly_graph = html.Div(
    [
        dcc.Graph(
            id="hourly",
            className="plot",
        )
    ],
    className="plot-container",
)

daily_graph = html.Div(
    [
        dcc.Graph(
            id="daily",
            className="plot",
        )
    ],
    className="plot-container",
)

scatter_graph = html.Div(
    [
        dcc.Graph(
            id="scatter",
            className="plot",
        )
    ],
    className="plot-container",
)
