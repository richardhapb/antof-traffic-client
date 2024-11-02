from waze.events import Events
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import requests
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

PERIM_X = [-70.42034224747098, -70.36743722434367]
PERIM_Y = [-23.721724880116387, -23.511242421131792]

PERIM_AFTA = gpd.GeoDataFrame(geometry=gpd.points_from_xy(PERIM_X, PERIM_Y))
PERIM_AFTA.crs = "EPSG:4326"
PERIM_AFTA = PERIM_AFTA.to_crs("EPSG:3857")

API_FERIADOS = "https://api.boostr.cl/holidays/2024.json"


def load_data(
    table_name: str,
    file: str = None,
    with_nested_items: bool = True,
    mode: str = "all",
    epoch: int = None,
    between: tuple = None,
):
    """
    Carga los datos de eventos de Waze desde un archivo JSON o desde la BD SQL
    """

    data = Events(table_name=table_name, filename=file)
    data.fetch_from_db(
        with_nested_items=with_nested_items, mode=mode, epoch=epoch, between=between
    )

    return data


def filter_location(dat: pd.DataFrame, x: list, y: list):
    """
    Filtra las coordenadas para Antofagasta, excluyendo las otras comunas
    """
    try:  # Alerts
        dat = dat[
            (dat["location"].apply(lambda loc: loc["x"] >= x[0] and loc["x"] <= x[1]))
        ]
        dat = dat[
            (dat["location"].apply(lambda loc: loc["y"] >= y[0] and loc["y"] <= y[1]))
        ]
    except KeyError:  # Jam
        dat = dat[
            (
                dat["line"].apply(
                    lambda line: line[0]["x"] >= x[0] and line[0]["x"] <= x[1]
                )
            )
        ]
        dat = dat[
            (
                dat["line"].apply(
                    lambda line: line[0]["y"] >= y[0] and line[0]["y"] <= y[1]
                )
            )
        ]

    return dat


def haversine(coordx: list, coordy: list):
    """
    Calcula la distancia entre dos coordenadas geográficas
    """
    lon1, lon2 = coordx
    lat1, lat2 = coordy

    # Convertir coordenadas de grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Diferencias de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.asin(np.sqrt(a))

    # Radio de la Tierra en kilómetros
    r = 6371

    # Distancia en metros
    return c * r * 1000


def freq_nearby(gdf, nearby_meters=200):
    """
    Cuenta cuántos puntos cercanos hay dentro de un radio especificado para cada punto en el GeoDataFrame.

    Parámetros:
    - gdf: GeoDataFrame que contiene geometrías de puntos.
    - radius: Radio de búsqueda en metros.

    Retorna:
    - GeoDataFrame con una columna adicional 'freq'.
    """
    # Asegura que el GeoDataFrame esté en un CRS proyectado con unidades en metros
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)

    # Extraer coordenadas x e y
    coords = np.vstack((gdf.geometry.x, gdf.geometry.y)).T

    # Construir el cKDTree
    tree = cKDTree(coords)

    # Consultar el árbol
    indices = tree.query_ball_point(coords, r=nearby_meters)

    # Contar vecinos (excluyendo el propio punto)
    neighbor_counts = [len(ind) - 1 for ind in indices]

    # Añadir los conteos al GeoDataFrame
    gdf["freq"] = neighbor_counts

    return gdf


def separate_coords(df):
    """
    Separa las coordenadas de un DataFrame en dos columnas, retornando un GeoDataFrame
    """
    df2 = df.copy()
    df2["x"] = df2["location"].apply(lambda x: x["x"])
    df2["y"] = df2["location"].apply(lambda y: y["y"])
    df2 = df2.drop(columns="location")
    df2["geometry"] = df2.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    dfg = gpd.GeoDataFrame(df2, geometry="geometry")

    # Establecer el sistema de referencia de coordenadas

    dfg = dfg.set_crs(epsg=4326)
    dfg = dfg.to_crs(epsg=3857)  # Adecuado para visualización en plano
    return dfg


def extract_event(data: gpd.GeoDataFrame, concept: list, extra_col: list = []):
    """
    Extraer los eventos de un tipo específico de un GeoDataFrame
    """

    dat = data.copy()

    if "geometry" not in dat.columns:
        dat = separate_coords(dat)

    dat = dat[dat["type"].isin(concept)][
        ["uuid", "street", "pubMillis", "endreport", "x", "y", "geometry"] + extra_col
    ]

    dat["pubMillis"] = pd.to_datetime(data["pubMillis"], unit="ms", utc=True)

    # Convertir la marca de tiempo a la zona horaria GMT-4 (CLT - Chile Standard Time)
    dat["pubMillis"] = dat["pubMillis"].dt.tz_convert("America/Santiago")

    # Ya se encuentra en GMT-4
    dat["endreport"] = pd.to_datetime(data["endreport"], unit="ms")
    dat["endreport"] = dat["endreport"].dt.tz_localize("America/Santiago")

    feriados = get_holidays()

    dat = dat.rename(columns={"pubMillis": "inicio", "endreport": "fin"})
    dat["hour"] = dat["inicio"].dt.hour
    dat["minute"] = dat["inicio"].dt.minute
    dat["day"] = dat["inicio"].dt.day
    dat["week_day"] = dat["inicio"].dt.dayofweek
    dat["day_type"] = dat["inicio"].apply(
        lambda x: "f"
        if (x.weekday() >= 5) | (x.strftime("%Y-%m-%d") in feriados)
        else "s"
    )
    return dat


def hourly_group(data: pd.DataFrame):
    """
    Transforma un DataFrame de eventos en un reporte por hora
    """

    df = data[["day_type", "hour", "inicio", "fin"]].copy()

    df.reset_index(inplace=True, drop=True)

    def calculate_hours(df):
        for i in range(df.shape[0]):
            hours = (df.loc[i, "fin"] - df.loc[i, "inicio"]).total_seconds() / 3600
            if np.isnan(hours):
                continue
            for h in range(1, int(hours) + 1):
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "day_type": [df.loc[i, "day_type"]],
                                "hour": [df.loc[i, "hour"] + h]
                                if (df.loc[i, "hour"] + h) < 24
                                else [df.loc[i, "hour"] + h - 24],
                                "inicio": [df.loc[i, "inicio"]],
                                "fin": [df.loc[i, "fin"]],
                            },
                        ),
                    ],
                    ignore_index=True,
                )
        return df

    df = calculate_hours(df)

    # Agrupar por hora y tipo de día
    hourly_reports = (
        df[["day_type", "hour"]].groupby(["day_type", "hour"]).size().unstack(level=0)
    )

    # Crear un índice que incluya todas las horas del día
    all_hours = pd.Index(range(24), name="hour")

    # Reindexar el DataFrame para incluir todas las horas del día
    hourly_reports = hourly_reports.reindex(all_hours, fill_value=0)

    return hourly_reports


def daily_group(data: pd.DataFrame):
    """
    Transforma un DataFrame de eventos en un reporte por día
    """

    df = data[["day_type", "day", "inicio", "fin"]].copy()

    df.reset_index(inplace=True, drop=True)

    def calculate_days(df):
        for i in range(df.shape[0]):
            days = (df.loc[i, "fin"] - df.loc[i, "inicio"]).days
            if np.isnan(days):
                continue
            for d in range(1, int(days) + 1):
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "day_type": [df.loc[i, "day_type"]],
                                "day": [df.loc[i, "day"] + d],
                                "inicio": [df.loc[i, "inicio"]],
                                "fin": [df.loc[i, "fin"]],
                            },
                        ),
                    ],
                    ignore_index=True,
                )
        return df

    df = calculate_days(df)

    # Agrupar por día y tipo de día
    daily_reports = (
        df[["day_type", "day"]].groupby(["day_type", "day"]).size().unstack(level=0)
    )

    # Crear un índice que incluya todos los días del mes
    all_days = pd.Index(range(1, 32), name="day")

    # Reindexar el DataFrame para incluir todos los días del mes
    daily_reports = daily_reports.reindex(all_days, fill_value=0)

    return daily_reports


def filter_nearby(gdf, threshold=300):
    """
    Filtra los puntos que tienen otros puntos cercanos dentro de un umbral dado.
    Retorna un GeoDataFrame con puntos únicos sin vecinos cercanos.

    Parámetros:
    - gdf: GeoDataFrame con geometrías de puntos.
    - threshold: Distancia umbral en metros.

    Retorna:
    - GeoDataFrame filtrado.
    """
    # Verificar y reproyectar si es necesario
    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=3857)

    # Extraer coordenadas x e y
    coords = np.vstack((gdf.geometry.x, gdf.geometry.y)).T

    # Construir el cKDTree
    tree = cKDTree(coords)

    # Inicializar un arreglo booleano para marcar los puntos a conservar
    n_points = len(gdf)
    keep = np.ones(n_points, dtype=bool)

    # Iterar sobre cada punto
    for idx in range(n_points):
        if keep[idx]:
            # Encontrar los índices de los puntos dentro del umbral
            indices = tree.query_ball_point(coords[idx], r=threshold)
            # Excluir el propio punto
            indices = [i for i in indices if i != idx]
            # Marcar los vecinos cercanos como False (eliminados)
            keep[indices] = False
        else:
            continue

    # Filtrar el GeoDataFrame
    gdf_filtered = gdf[keep].reset_index(drop=True)
    return gdf_filtered


def plot_map(
    data: gpd.GeoDataFrame,
    title: str,
    cmap: str = "viridis",
    markersize: int = 10,
    ax: plt.axes = None,
    figsize: tuple = (4.5, 9.5),
):
    """
    Grafica un GeoDataFrame en un eje de coordenadas
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)

    sc = data.to_crs(epsg=3857).geometry.plot(
        ax=ax, cmap=cmap, alpha=0.8, markersize=markersize, c=data.freq
    )
    ax.set_xlim(PERIM_AFTA.total_bounds[0], PERIM_AFTA.total_bounds[2])
    ax.set_ylim(PERIM_AFTA.total_bounds[1], PERIM_AFTA.total_bounds[3])
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    cbar = sc.get_figure().colorbar(sc.collections[0], ax=ax)
    cbar.set_label("Frecuencia")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    plt.xticks(rotation=45)
    ax.set_title(title)
    return fig, ax


def get_holidays():
    feriados = requests.get(API_FERIADOS).json()

    feriados = feriados["data"]
    feriados = [f["date"] for f in feriados]

    return feriados


def grid(geometry: gpd.GeoDataFrame, n_x_div: int, n_y_div: int):
    bounds_x = np.array(
        np.linspace(
            geometry.to_crs(epsg=3857).geometry.x.min(),
            geometry.to_crs(epsg=3857).geometry.x.max(),
            n_x_div,
        )
    )
    bounds_y = np.array(
        np.linspace(
            geometry.to_crs(epsg=3857).geometry.y.min(),
            geometry.to_crs(epsg=3857).geometry.y.max(),
            n_y_div,
        )
    )

    return np.meshgrid(bounds_x, bounds_y)


def calc_quadrant(x_pos: int, y_pos: int, x_len: int):
    return x_len * y_pos + x_pos + 1


def get_quadrant(x_grid: np.array, y_grid: np.array, point: tuple):
    x_pos, y_pos = -1, -1

    for xi in range(len(x_grid[0])):
        if (
            xi < len(x_grid[0]) - 1
            and point[0] >= x_grid[0][xi]
            and point[0] <= x_grid[0][xi + 1]
        ):
            x_pos = xi

    for yi in range(len(y_grid)):
        if (
            yi < len(y_grid) - 1
            and point[1] >= y_grid[yi][0]
            and point[1] <= y_grid[yi + 1][0]
        ):
            y_pos = yi

    if x_pos < 0 or y_pos < 0:
        raise ValueError(f"El punto {point} no se encuentra en ningún cuadrante")

    quadrant = x_pos, y_pos

    return quadrant


def plot_antof():
    PERIM_AFTA = gpd.GeoDataFrame(geometry=gpd.points_from_xy(PERIM_X, PERIM_Y))
    PERIM_AFTA.crs = "EPSG:4326"
    PERIM_AFTA = PERIM_AFTA.to_crs(epsg=3857)

    fig, ax = plt.subplots()

    fig.set_size_inches(10, 10)
    PERIM_AFTA.plot(ax=ax, color="red")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_xlim(PERIM_AFTA.total_bounds[0], PERIM_AFTA.total_bounds[2])
    ax.set_ylim(PERIM_AFTA.total_bounds[1], PERIM_AFTA.total_bounds[3])
    return ax


def get_center_points(grid: tuple):
    # X
    center_points_x = np.zeros((grid[0].shape[0] - 1, grid[0].shape[1] - 1))

    x_half = (grid[0][0][1] - grid[0][0][0]) / 2

    for x in range(len(grid[0][0]) - 1):
        center_points_x[0][x] = grid[0][0][x] + x_half

    center_points_x[:][:] = center_points_x[0][:]

    # Y
    center_points_y = np.zeros((grid[0].shape[0] - 1, grid[0].shape[1] - 1))

    y_half = (grid[1][1][0] - grid[1][0][0]) / 2

    for y in range(len(grid[1]) - 1):
        center_points_y[y][0] = grid[1][y][0] + y_half

    for c in range(len(center_points_y)):
        center_points_y[c][:] = center_points_y[c][0]

    return center_points_x, center_points_y


def xgb_classifier(
    df: gpd.GeoDataFrame,
    Y_label: str,
    cats: list = [],
    ohe: bool = True,
    happen: bool = True,
):
    events = df.copy()

    if happen:
        events["happen"] = 1

    q_events = len(events)
    if "street" in df.columns:
        street = events["street"]
    if "day_type" in df.columns:
        day_type = np.random.choice(["s", "f"], q_events)
    if "type" in df.columns:
        type = np.random.choice(["ACCIDENT", "JAM"], q_events)
    if "group" in df.columns:
        group = np.random.randint(df["group"].min(), df["group"].max(), q_events)
    if "day" in df.columns:
        day = np.random.randint(events.day.min(), events.day.max() + 1, q_events)
    if "week_day" in df.columns:
        week_day = np.random.randint(
            events.week_day.min(), events.week_day.max(), q_events
        )
    if "hour" in df.columns:
        hour = np.random.randint(events.hour.min(), events.hour.max(), q_events)
    if "minute" in df.columns:
        minute = np.random.randint(events.minute.min(), events.minute.max(), q_events)

    no_events = pd.DataFrame(columns=events.columns)

    if "street" in df.columns:
        no_events["street"] = street
    if "day_type" in df.columns:
        no_events["day_type"] = day_type
    if "type" in df.columns:
        no_events["type"] = type
    if "group" in df.columns:
        no_events["group"] = group
    if "day" in df.columns:
        no_events["day"] = day
    if "week_day" in df.columns:
        no_events["week_day"] = week_day
    if "hour" in df.columns:
        no_events["hour"] = hour
    if "minute" in df.columns:
        no_events["minute"] = minute

    if happen:
        no_events["happen"] = 0

    total_events = pd.concat([events, no_events], axis=0)

    if "day_type" in total_events.columns:
        total_events["day_type"] = total_events["day_type"].map({"f": 0, "s": 1})

    if ohe:
        labeled = []
        for c in cats:
            onehot = OneHotEncoder(handle_unknown="ignore")
            oht = onehot.fit_transform(total_events[[c]])

            labeled.append(
                pd.DataFrame(
                    oht.toarray(), columns=onehot.get_feature_names_out()
                ).reset_index(drop=True)
            )

        X_labeled = pd.concat(
            [total_events.drop(cats, axis=1).reset_index(drop=True), *labeled], axis=1
        )
    else:
        X_labeled = total_events.copy()

    if happen:
        Y = X_labeled["happen"]
        X_labeled = X_labeled.drop("happen", axis=1)
    else:
        Y = X_labeled[Y_label]
        X_labeled = X_labeled.drop(Y_label, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, Y, test_size=0.2, random_state=42
    )

    xgb = XGBClassifier(
        learning_rate=0.03,
        random_state=42,
        n_estimators=50,
        max_depth=5,
        gamma=0.2,
        colsample_bytree=0.7,
    )
    xgb.fit(X_train, y_train)

    cvs = cross_val_score(xgb, X_labeled, Y, cv=10)
    print(cvs)
    print(cvs.mean())

    return {
        "model": xgb,
        "X_labeled": X_labeled,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
