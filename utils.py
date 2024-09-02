import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import json


PERIM_X = [-70.45534224747098, -70.32743722434367]
PERIM_Y = [-23.701724880116387, -23.411242421131792]

def load_data(file:str='data.json'):
    '''
    Carga un archivo JSON en un DataFrame
    '''
    with open(file, "r", encoding='UTF8') as f:
        data = json.load(f)
        f.close()

    alerts = pd.DataFrame(data['alerts'])
    jams = pd.DataFrame(data['jams'])

    return alerts, jams

def filter_location(dat: pd.DataFrame,x:list, y:list):
    '''
    Filtra las coordenadas para Antofagasta, excluyendo las otras comunas
    '''
    try: # Alerts
        dat = dat[(dat['location'].apply(lambda loc: loc['x'] >= x[0] and loc['x'] <= x[1] ))]
        dat = dat[(dat['location'].apply(lambda loc: loc['y'] >= y[0] and loc['y'] <= y[1] ))]
    except KeyError: # Jam
        dat = dat[(dat['line'].apply(lambda line: line[0]['x'] >= x[0] and line[0]['x'] <= x[1]))]
        dat = dat[(dat['line'].apply(lambda line: line[0]['y'] >= y[0] and line[0]['y'] <= y[1]))]

    return dat

def haversine(coordx:list, coordy:list):
    '''
    Calcula la distancia entre dos coordenadas geográficas
    '''
    lon1, lon2 = coordx
    lat1, lat2 = coordy

    # Convertir coordenadas de grados a radianes
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Diferencias de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula de Haversine
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.asin(np.sqrt(a))

    # Radio de la Tierra en kilómetros
    r = 6371

    # Distancia en metros
    return c * r * 1000

def nearby(row, df, nearby_meters=200):
    '''
    Cuenta cuántos puntos hay cerca de un punto dado
    '''
    c = 0
    for i in df.index:
        c += 1 if row.geometry.distance(df.loc[i, 'geometry']) <= nearby_meters else 0
    return c

def separate_coords(df):
    '''
    Separa las coordenadas de un DataFrame en dos columnas, retornando un GeoDataFrame
    '''
    df2 = df.copy()
    df2['x'] = df2['location'].apply(lambda x: x['x'])
    df2['y'] = df2['location'].apply(lambda y: y['y'])
    df2 = df2.drop(columns="location")
    df2['geometry'] = df2.apply(lambda row: Point(row['x'], row['y']), axis=1)
    dfg = gpd.GeoDataFrame(df2, geometry='geometry')

    # Establecer el sistema de referencia de coordenadas

    dfg = dfg.set_crs(epsg=4326)
    dfg = dfg.to_crs(epsg=3857) # Adecuado para visualización en plano
    return dfg

def freq_nearby(df:gpd.GeoDataFrame, nearby_meters:float=200):
    '''
    Calcula la cantidad de puntos cercanos a cada punto
    '''
    freq = df.apply(lambda row: nearby(row, df, nearby_meters), axis=1)

    return freq

def extract_event(data:gpd.GeoDataFrame, concept:str):
    '''
    Extraer los eventos de un tipo específico de un GeoDataFrame
    '''
    dat = data[data['type'] == concept][['uuid', 'street', 'pubMillis', 'endreport', 'x', 'y', 'geometry']]

    dat['pubMillis'] = pd.to_datetime(data['pubMillis'], unit='ms', utc=True)

    # Convertir la marca de tiempo a la zona horaria GMT-4 (CLT - Chile Standard Time)
    dat['pubMillis'] = dat['pubMillis'].dt.tz_convert('America/Santiago')
    
    # Ya se encuentra en GMT-4
    dat['endreport'] = pd.to_datetime(data['endreport'], unit='ms')
    
    dat = dat.rename(columns={'pubMillis': 'inicio', 'endreport':'fin'})
    dat['hour'] = dat['inicio'].dt.hour
    dat['day'] = dat['inicio'].dt.dayofweek
    dat['day_type'] = dat['day'].apply(lambda x: 'Semana' if x < 5 else 'Fin de semana')    
    return dat

def hourly_group(data:pd.DataFrame):
    '''
    Transforma un DataFrame de eventos en un reporte por hora
    '''
    # Agrupar por hora y tipo de día
    hourly_reports = data.groupby(['day_type', 'hour']).size().unstack(level=0)

        # Crear un índice que incluya todas las horas del día
    all_hours = pd.Index(range(24), name='hour')

    # Reindexar el DataFrame para incluir todas las horas del día
    hourly_reports = hourly_reports.reindex(all_hours, fill_value=0)

    return hourly_reports

def filter_nearby(data:gpd.GeoDataFrame, nearby_meters:float=200):
    '''
    Filtra los puntos cercanos a una distancia dada
    '''
    unique = [True] * len(data)
    for i in range(0, len(data)):
        for j in range(i+1, len(data)):
            if not unique[j]:
                continue
            dist = haversine([data.iloc[i].x, data.iloc[j].x], [data.iloc[i].y, data.iloc[j].y])
            if dist < nearby_meters:
                unique[j] = False
    return unique
