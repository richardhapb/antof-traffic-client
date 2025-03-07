import json
import warnings
from datetime import datetime
from typing import List

from geopandas.geodataframe import GeoDataFrame
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import requests

from waze.events import Events

# Deshabilitar SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

TZ = "America/Santiago"

PERIM_X = [-70.42034224747098, -70.36743722434367]
PERIM_Y = [-23.721724880116387, -23.511242421131792]

PERIM_AFTA = gpd.GeoDataFrame(geometry=gpd.points_from_xy(PERIM_X, PERIM_Y))
PERIM_AFTA.crs = "EPSG:4326"
PERIM_AFTA = PERIM_AFTA.to_crs("EPSG:3857")

API_FERIADOS = "https://api.boostr.cl/holidays/{year}.json"


def load_data(
    table_name: str,
    file: str | None = None,
    with_nested_items: bool = True,
    mode: str = "all",
    epoch: int | None = None,
    between: tuple | None = None,
) -> Events:
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


def update_timezone(
    data: pd.DataFrame, tz: str = TZ
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Actualiza el timezone de los datos de eventos
    """

    data_copy = data.copy()

    data_copy["pubMillis"] = pd.to_datetime(data_copy["pubMillis"], unit="ms", utc=True)
    data_copy["pubMillis"] = data_copy["pubMillis"].dt.tz_convert(tz)
    data_copy["endreport"] = pd.to_datetime(data_copy["endreport"], unit="ms", utc=True)
    data_copy["endreport"] = data_copy["endreport"].dt.tz_convert(tz)

    return data_copy


def freq_nearby(gdf: gpd.GeoDataFrame, nearby_meters=200) -> gpd.GeoDataFrame:
    """
    Cuenta cuántos puntos cercanos hay dentro de un radio especificado para cada punto en el GeoDataFrame.

    Parámetros:
    - gdf: GeoDataFrame que contiene geometrías de puntos.
    - radius: Radio de búsqueda en metros.

    Retorna:
    - GeoDataFrame con una columna adicional 'freq'.
    """
    gdf2 = gdf.copy()
    assert gdf2.crs is not None, "GeoDataFrame vacío"
    # Asegura que el GeoDataFrame esté en un CRS proyectado con unidades en metros
    if gdf2.crs.is_geographic:
        gdf2 = gdf2.to_crs(epsg=3857)

    if gdf2 is None:
        return gpd.GeoDataFrame()
    # Extraer coordenadas x e y
    coords = np.vstack((gdf2.geometry.x, gdf2.geometry.y)).T

    # Construir el cKDTree
    tree = cKDTree(coords)

    # Consultar el árbol
    indices = tree.query_ball_point(coords, r=nearby_meters)

    # Contar vecinos (excluyendo el propio punto)
    neighbor_counts = [len(ind) - 1 for ind in indices]

    # Añadir los conteos al GeoDataFrame
    gdf2["freq"] = neighbor_counts

    return gdf2


def separate_coords(df: pd.DataFrame) -> GeoDataFrame:
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
    dfg = dfg.to_crs(epsg=3857)  # Adecuado para visualización en mapa

    if dfg is None:
        dfg = gpd.GeoDataFrame()

    return dfg


def extract_event(
    data: gpd.GeoDataFrame | pd.DataFrame, concept: list, extra_col: list
) -> gpd.GeoDataFrame:
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

    holidays = get_holidays()

    dat["hour"] = dat["inicio"].dt.hour
    dat["minute"] = dat["inicio"].dt.minute
    dat["day"] = dat["inicio"].dt.day
    dat["week_day"] = dat["inicio"].dt.dayofweek
    dat["day_type"] = dat["inicio"].apply(
        lambda x: (
            "f" if (x.weekday() >= 5) | (x.strftime("%Y-%m-%d") in holidays) else "s"
        )
    )
    try:
        dat = dat.loc[dat["type"].isin(concept)][millis + extra_col]
    except KeyError:
        dat = dat.loc[dat["type"].isin(concept)][[millis[0]] + extra_col]

    return dat


def hourly_group(
    data: pd.DataFrame | gpd.GeoDataFrame, do_sum: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Transforma un DataFrame de eventos en un reporte por hora
    """

    df = data[["day_type", "hour", "inicio", "fin"]].copy()

    df.reset_index(inplace=True, drop=True)

    days = (data["inicio"].max() - data["inicio"].min()).days

    days = 1 if days <= 0 else days

    # Agrupar por hora y tipo de día
    hourly_reports = (
        df.loc[:, ["day_type", "hour"]]
        .groupby(["day_type", "hour"])
        .size()
        .unstack(level=0)
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

    if not do_sum:
        # Calcular la tasa por tipo de día
        hourly_reports["f"] = hourly_reports["f"] / days
        hourly_reports["s"] = hourly_reports["s"] / days

    return hourly_reports


def daily_group(
    data: pd.DataFrame | gpd.GeoDataFrame,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Transforma un DataFrame de eventos en un reporte por día
    """

    df = data[["day_type", "day", "inicio", "fin"]].copy()

    df.reset_index(inplace=True, drop=True)

    # Agrupar por día y tipo de día
    daily_reports = (
        df.loc[:, ["day_type", "day"]]
        .groupby(["day_type", "day"])
        .size()
        .unstack(level=0)
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


def get_holidays() -> List:
    """
    Retorna listado de feriados
    Args:
    :param None: No parameters.
    Returns:
        Return list: Listado de feriados en formato dd-mm-YY
    """
    years = [year for year in range(2024, datetime.now().year + 1)]
    holidays = []

    try:
        try:
            for year in years:
                url = API_FERIADOS.format(year=year)
                response = requests.get(url, timeout=10).json()['data']
                holidays.extend(response)

            holidays = [f["date"] for f in holidays]

            with open("data/holidays.json", "w", encoding="utf-8") as fp:
                json.dump({"holidays": holidays}, fp)
        except (ConnectionError, TimeoutError) as e:
            print(
                "Error al conectar para obtener feriados, cargando archivo de respaldo"
            )
            print(f"ERROR: {e}")
            with open("data/holidays.json", "r", encoding="utf-8") as f:
                holidays = json.load(f)["holidays"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error al leer el archivo")
        print(f"ERROR: {e}")
        holidays = []

    return holidays
