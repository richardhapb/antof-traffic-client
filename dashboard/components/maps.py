from dash import html, dcc

main_map = html.Div(
    id="map-zone",
    children=[
        dcc.Graph(id="map", className="plot"),
    ],
    className="plot-container",
)

ml_map = html.Div(
    [
        dcc.Graph(
            id="ML",
            className="ml-map",
            config={"scrollZoom": True},
        )
    ],
    className="plot-container ml-zone",
)
