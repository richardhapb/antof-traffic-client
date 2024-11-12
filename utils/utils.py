from waze.events import Events
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import contextily as cx
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import requests
import json
import warnings

# Deshabilitar SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

tz = "America/Santiago"

PERIM_X = [-70.42034224747098, -70.36743722434367]
PERIM_Y = [-23.721724880116387, -23.511242421131792]

PERIM_AFTA = gpd.GeoDataFrame(geometry=gpd.points_from_xy(PERIM_X, PERIM_Y))
PERIM_AFTA.crs = "EPSG:4326"
PERIM_AFTA = PERIM_AFTA.to_crs("EPSG:3857")

API_FERIADOS = "https://api.boostr.cl/holidays.json"


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

    Parámetros:
    - table_name: Nombre de la tabla de la base de datos
    - file: Archivo JSON con los datos de eventos
    - with_nested_items: Si se deben cargar los eventos con sus subeventos
    - mode: Modo de carga de datos, puede ser "all", "last_24h", "since" o "between"
    - epoch: Hora de inicio de la búsqueda, en formato timestamp UTC en milisegundos
    - between: Rango de tiempo de búsqueda, en formato timestamp UTC en milisegundos
    """

    events = Events(table_name=table_name, filename=file)
    events.fetch_from_db(
        with_nested_items=with_nested_items, mode=mode, epoch=epoch, between=between
    )

    return events


def update_timezone(data: pd.DataFrame, tz: str = tz):
    """
    Actualiza el timezone de los datos de eventos
    """

    data_copy = data.copy()

    data_copy["pubMillis"] = pd.to_datetime(data_copy["pubMillis"], unit="ms", utc=True)
    data_copy["pubMillis"] = data_copy["pubMillis"].dt.tz_convert(tz)
    data_copy["endreport"] = pd.to_datetime(data_copy["endreport"], unit="ms", utc=True)
    data_copy["endreport"] = data_copy["endreport"].dt.tz_convert(tz)

    return data_copy


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

    if "geometry" not in dat.columns and "geometry" in extra_col:
        dat = separate_coords(dat)

    millis = (
        ["inicio", "fin"]
        if "inicio" in list(data.columns)
        else ["pubMillis", "endreport"]
    )
    try:
        if "pubMillis" in millis:
            dat = dat.rename(columns={"pubMillis": "inicio", "endreport": "fin"})
    except KeyError:
        if "pubMillis" in millis:
            dat = dat.rename(columns={"pubMillis": "inicio"})

    millis = ["inicio", "fin"]

    try:
        try:
            feriados = get_holidays()
            with open("data/holidays.json", "w") as fp:
                json.dump({"feriados": feriados}, fp)
        except ConnectionError or TimeoutError:
            print(
                "Error al conectar para obtener feriados, cargando archivo de respaldo"
            )
            with open("data/holidays.json", "r") as f:
                feriados = json.load(f)["feriados"]
    except FileNotFoundError or json.JSONDecodeError:
        print("Error al leer el archivo")
        feriados = []

    dat["hour"] = dat["inicio"].dt.hour
    dat["minute"] = dat["inicio"].dt.minute
    dat["day"] = dat["inicio"].dt.day
    dat["week_day"] = dat["inicio"].dt.dayofweek
    dat["day_type"] = dat["inicio"].apply(
        lambda x: "f"
        if (x.weekday() >= 5) | (x.strftime("%Y-%m-%d") in feriados)
        else "s"
    )
    try:
        dat = dat[dat["type"].isin(concept)][millis + extra_col]
    except KeyError:
        dat = dat[dat["type"].isin(concept)][[millis[0]] + extra_col]

    return dat


def hourly_group(data: pd.DataFrame, sum: bool = False):
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

    # df = calculate_hours(df)

    days = (data["inicio"].max() - data["inicio"].min()).days

    days = 1 if days <= 0 else days

    # Agrupar por hora y tipo de día
    hourly_reports = (
        df[["day_type", "hour"]].groupby(["day_type", "hour"]).size().unstack(level=0)
    )

    # Crear un índice que incluya todas las horas del día
    all_hours = pd.Index(range(24), name="hour")

    # Reindexar el DataFrame para incluir todas las horas del día
    hourly_reports = hourly_reports.reindex(all_hours, fill_value=0)
    hourly_reports = hourly_reports.fillna(0)
    hourly_reports = hourly_reports.reset_index()

    if "s" not in hourly_reports.columns:
        hourly_reports["s"] = 0.0
    else:
        hourly_reports["s"] = hourly_reports["s"].astype(float)

    if "f" not in hourly_reports.columns:
        hourly_reports["f"] = 0.0
    else:
        hourly_reports["f"] = hourly_reports["f"].astype(float)

    if not sum:
        # Calcular la tasa por tipo de día
        hourly_reports["f"] = hourly_reports["f"] / days
        hourly_reports["s"] = hourly_reports["s"] / days

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

    # df = calculate_days(df)

    # Agrupar por día y tipo de día
    daily_reports = (
        df[["day_type", "day"]].groupby(["day_type", "day"]).size().unstack(level=0)
    )

    # Crear un índice que incluya todos los días del mes
    all_days = pd.Index(range(1, 32), name="day")

    start_date = data["inicio"].min()
    end_date = data["inicio"].max()
    months = (end_date.year - start_date.year) * 12 + (
        end_date.month - start_date.month
    )

    months = 1 if months <= 0 else months
    # Reindexar el DataFrame para incluir todos los días del mes
    daily_reports = daily_reports.reindex(all_days, fill_value=0)

    if "s" not in daily_reports.columns:
        daily_reports["s"] = 0.0
    else:
        daily_reports["s"] = daily_reports["s"].astype(float)

    if "f" not in daily_reports.columns:
        daily_reports["f"] = 0.0
    else:
        daily_reports["f"] = daily_reports["f"].astype(float)

    # Calcular la tasa por tipo de día
    daily_reports["f"] = daily_reports["f"] / months
    daily_reports["s"] = daily_reports["s"] / months
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
