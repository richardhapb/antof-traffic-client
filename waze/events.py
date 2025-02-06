
# Typing
from typing import List, Dict, Any, Set

# Datetime
from datetime import datetime
import pytz

# Filesystem
import shutil
import os

# DATABASE
from config import DATABASE_CONFIG
import psycopg2
from psycopg2.extensions import connection

# Decorator
from functools import wraps

# data
import pandas as pd
import json
import geopandas as gpd

def db_connection(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.table_name is None or self.table_name == "":
            raise ValueError("No se ha especificado la tabla de la base de datos")
        if self.db is None or self.db.closed:
            self.db = psycopg2.connect(**DATABASE_CONFIG)
            self.db.set_client_encoding("UTF8")

        try:
            result = None
            if self.db:
                result = func(self, *args, **kwargs)
        except psycopg2.Error as e:
            self.db.rollback()
            raise ValueError(f"Error executing database query: {e}")
        else:
            self.db.commit()
            return result
        finally:
            self.db.close()
            self.db = None

    return wrapper


class Events:
    # Mapeo de columnas de Waze a columnas de la base de datos
    # Se usa para insertar datos en la base de datos
    # Cada tabla tiene diccionarios dentro, para esos casos se considera [nombre_atributo]_id
    # Además existe una tabla para cada diccionario dentro de los datos, con nombre [nombre_tabla]_[nombre_atributo]

    db_columns_map : Dict = {
        "alerts": {
            "uuid": "uuid",
            "reliability": "reliability",
            "type": "type",
            "roadType": "road_type",
            "magvar": "magvar",
            "subtype": "subtype",
            "location": "location_id",
            "street": "street",
            "pubMillis": "pub_millis",
            "endreport": "end_pub_millis",
        },
        "jams": {
            "uuid": "uuid",
            "level": "level",
            "speedKMH": "speed_kmh",
            "length": "length",
            "endNode": "end_node",
            "roadType": "road_type",
            "delay": "delay",
            "street": "street",
            "line": {
                "position": "position",
                "x": "x",
                "y": "y",
            },
            "segments": {
                "position": "position",
                "ID": "segment_id",
                "fromNode": "from_node",
                "toNode": "to_node",
                "isForward": "is_forward",
            },
            "pubMillis": "pub_millis",
            "endreport": "end_pub_millis",
        },
    }

    def __init__(self, data:List[Dict[str, Any]]=[], filename:str | None=None, table_name:str | None=None)->None:
        self.data:List[Dict[str, Any]] = data
        self.filename:str | None = filename
        self.table_name:str | None = table_name
        self.pending_endreports: Set = set()
        self.db: connection | None = None
        self.index_map: Dict = {}

        if self.data is not None:
            self.update_index_map()
            self.update_pending_endreports()

    def __del__(self)->None:
        if self.db is not None and not self.db.closed:
            self.db.close()

    def __add__(self, other: "Events")->"Events":
        if not isinstance(other, Events):
            raise TypeError("Operador + no soportado")
        if other.data is not None and len(other.data) > 0:
            new_data = [
                d for d in other.data if d["uuid"] not in self.pending_endreports
            ]
        elif self.data is not None and len(self.data) > 0:
            new_data = [
                d for d in self.data if d["uuid"] not in other.pending_endreports
            ]
        else:
            return Events([])

        events = Events(
            new_data + self.data,
            self.filename,
            self.table_name if self.table_name else other.table_name,
        )

        events.pending_endreports = (
            self.pending_endreports | other.pending_endreports
        )

        # Update pending endreports if an uuid is not found in the new data
        for uuid in self.pending_endreports:
            if uuid not in other.pending_endreports:
                events.end_report(uuid)

        return events

    def __sub__(self, other:"Events")->"Events":
        if not isinstance(other, Events):
            raise TypeError("Operador - no soportado")
        if other.data is not None and len(other.data) > 0:
            new_data = [
                d for d in self.data if d["uuid"] not in other.pending_endreports
            ]
        else:
            new_data = self.data

        events = Events(
            new_data,
            self.filename if self.filename else other.filename,
            self.table_name if self.table_name else other.table_name,
        )

        events.pending_endreports = (
            self.pending_endreports - other.pending_endreports
        )

        return events

    def update_index_map(self)->None:
        if self.data is not None:
            self.index_map = {d["uuid"]: i for i, d in enumerate(self.data)}
        else:
            self.index_map = {}

    def read_file(self, filename:str | None=None) -> List[Dict[str, Any]]:
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para leer el archivo")

        if filename is not None:
            self.filename = filename
        else:
            return []
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Archivo {self.filename} no existe")

        try:
            with open(self.filename, "r") as f:
                self.data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decodificando archivo {self.filename}: {e}")
            print("Se utilizará backup")
            try:
                with open(self.filename + ".bak", "r") as f:
                    self.data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decodificando backup: {e}")
                raise e

        self.update_index_map()
        self.update_pending_endreports()
        return self.data

    def write_file(self, filename:str | None=None)->None:
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para crear el archivo")

        if filename is not None:
            self.filename = filename
        else:
            return

        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))

        if os.path.exists(self.filename):
            shutil.copyfile(self.filename, self.filename + ".bak")

        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def update_pending_endreports(self)->None:
        if self.data is None:
            return
        self.pending_endreports = {
            d["uuid"]
            for d in self.data
            if "endreport" not in d or d["endreport"] is None
        }

    def end_report(self, uuid:str)->None:
        now = int((datetime.now(tz=pytz.utc).timestamp() - (5 * 60 / 2)) * 1000)

        idx = self.index_map.get(uuid, None)

        if idx is not None:
            self.data[idx]["endreport"] = now

            if self.table_name != "" and self.table_name is not None:
                self.update_endreport_to_db(now, uuid)

            self.pending_endreports.discard(uuid)

    def clean_data(self)->None:
        if self.data is None or self.table_name is None:
            return
        deletions = set(k for d in self.data for k in d) - set(
            self.db_columns_map[self.table_name].keys()
        )
        for item in self.data:
            for key in deletions:
                if key in item:
                    item.pop(key, None)

    @staticmethod
    def format_data(data: List[tuple[Any]], table_name:str)->List[Dict[str, Any]]:
        if data is None or len(data) == 0:
            return []
        if not isinstance(data[0], tuple):
            return []
        if not table_name:
            return []
        if data is not None:
            cols = [
                k
                for k in Events.db_columns_map[table_name].keys()
                if k not in ["line", "segments"]
            ]
            events = [{k: v for k, v in zip(cols, d)} for d in data]

        return events

    def to_gdf(self, tz: str = "America/Santiago")->gpd.GeoDataFrame:
        from utils import utils
        if len(self.data) == 0:
            return gpd.GeoDataFrame()
        df = pd.DataFrame(self.data, columns=list(self.data[0].keys())) # type: ignore
        df = utils.update_timezone(df, tz)

        return utils.separate_coords(df)

    def fetch_from_db(
            self, mode:str="last_24h", with_nested_items:bool=False, epoch:int | None=None, between:tuple | None=None
    ):
        data = self.get_all_from_db(mode=mode, epoch=epoch, between=between)

        if data is None:
            return

        self.data = data

        self.update_index_map()
        self.update_pending_endreports()
        if with_nested_items:
            self.fetch_nested_items()

    @db_connection
    def fetch_nested_items(self):
        sqls = []
        if self.table_name == "alerts":
            sql = (
                f"SELECT a.*, l.x, l.y FROM {self.table_name} a "
                f"JOIN {self.table_name}_location l ON a.location_id = l.id"
            )
            sqls.append(sql)
        elif self.table_name == "jams":
            sql = (
                f"SELECT s.* FROM {self.table_name} j "
                f"JOIN {self.table_name}_segments s ON j.uuid = s.{self.table_name}_uuid "
                f"ORDER BY s.position"
            )
            sqls.append(sql)
            sql = (
                f"SELECT l.* FROM {self.table_name} j "
                f"JOIN {self.table_name}_line l ON j.uuid = l.{self.table_name}_uuid"
            )
            sqls.append(sql)
        else:
            return

        records = []

        assert isinstance(self.db, connection), "Error in db connection"
        
        cur = self.db.cursor()
        for sql in sqls:
            cur.execute(sql)
            record = cur.fetchall()
            if self.table_name == "jams":
                record = [r for r in record if r[1] in self.index_map]

            records.append(record)
        cur.close()

        if self.table_name == "alerts":
            for record in records[0]:
                if record[0] not in self.index_map:
                    continue
                self.data[self.index_map[record[0]]]["location"] = {
                    "x": record[-2],
                    "y": record[-1],
                }
        elif self.table_name == "jams":
            for record in records[0]:
                if "segments" not in self.data[self.index_map[record[1]]]:
                    self.data[self.index_map[record[1]]]["segments"] = []
                self.data[self.index_map[record[1]]]["segments"] += [
                    {
                        k: v
                        for k, v in zip(
                            self.db_columns_map[self.table_name]["segments"].keys(),
                            record[2:],
                        )
                    }
                ]
            for record in records[1]:
                if "line" not in self.data[self.index_map[record[1]]]:
                    self.data[self.index_map[record[1]]]["line"] = []
                self.data[self.index_map[record[1]]]["line"] += [
                    {
                        k: v
                        for k, v in zip(
                            self.db_columns_map[self.table_name]["line"].keys(),
                            record[2:],
                        )
                    }
                ]

    @db_connection
    def get_all_from_db(
        self, mode: str = "last_24h", epoch: int | None = None, between: tuple | None = None
    ):
        if self.table_name is None:
            return
        not_ended = "not_ended" in mode
        last_24h = "last_24h" in mode
        since = "since" in mode and epoch is not None
        is_between = "between" in mode and between is not None
        all = "all" in mode

        events = []

        assert isinstance(self.db, connection), "Error in db connection"
        cur = self.db.cursor()
        if not_ended:
            cur.execute(
                "SELECT * FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            events = cur.fetchall()

        if last_24h:
            cur.execute(
                "SELECT * FROM " + self.table_name + " WHERE pub_millis > %s",
                (int((datetime.now(tz=pytz.utc).timestamp() - (24 * 60 * 60)) * 1000),),
            )
            events = cur.fetchall()

        if since:
            cur.execute(
                "SELECT * FROM " + self.table_name + " WHERE pub_millis > %s",
                (epoch,),
            )
            events = cur.fetchall()

        if is_between:
            cur.execute(
                "SELECT * FROM "
                + self.table_name
                + " WHERE pub_millis BETWEEN %s AND %s",
                between,
            )
            events = cur.fetchall()

        if all:
            cur.execute("SELECT * FROM " + self.table_name)
            events = cur.fetchall()

        cur.close()
       
        data = Events.format_data(events, self.table_name)
        return data

    @db_connection
    def insert_to_db(self, review_mode:str="last_24h")->None:
        if self.table_name is None:
            return
        only_review_not_ended = "not_ended" in review_mode
        only_review_last_24h = "last_24h" in review_mode
        review_all = "all" in review_mode

        assert isinstance(self.db, connection), "Error in db connection"
        cur = self.db.cursor()

        if only_review_not_ended:
            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            not_ended = {d[0] for d in cur.fetchall()}
        elif only_review_last_24h:
            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE pub_millis > %s AND end_pub_millis IS NULL",
                (int((datetime.now(tz=pytz.utc).timestamp() - (24 * 60 * 60)) * 1000),),
            )
            not_ended = {d[0] for d in cur.fetchall()}
        elif review_all:
            cur.execute("SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL")
            not_ended = {d[0] for d in cur.fetchall()}
        else:
            not_ended = set()

        # Lista de uuids que no se encuentran en la base de datos
        data = [self.data[i] for d, i in self.index_map.items() if d not in not_ended]
        print(f"Insertando {len(data)} elementos en {self.table_name}")
        for record in data:
            # Identificar si estamos en el caso de `alerts` o `jams`
            if self.table_name == "alerts":
                # Caso `alerts`: Crear referencia `location_id` para `alerts_location`
                location_data = record.pop(
                    "location", None
                )  # Extraer el diccionario de `location`
                if location_data:
                    # Insertar datos en `alerts_location` y obtener el `location_id`
                    sql_location = "INSERT INTO alerts_location (x, y) VALUES (%s, %s) RETURNING id"
                    cur.execute(
                        sql_location,
                        (
                            location_data["x"],
                            location_data["y"],
                        ),
                    )
                    result = cur.fetchone()
                    if result is not None:
                        location_id = result[0]  # Obtener el `location_id` generado

                        # Agregar `location_id` a `record` para la inserción en `alerts`
                        record["location"] = location_id

                # Insertar en la tabla `alerts`
                columns = ", ".join(
                    [
                        f"{self.db_columns_map[self.table_name][k]}"
                        for k in record.keys()
                    ]
                )
                placeholders = ", ".join(["%s"] * len(record))
                sql_alerts = f"INSERT INTO alerts ({columns}) VALUES ({placeholders})"
                cur.execute(sql_alerts, tuple(record.values()))

            elif self.table_name == "jams":
                # Caso `jams`: `jams_segments` y `jams_lines` hacen referencia a `jams` con `jam_uuid`
                jam_uuid = record["uuid"]  # Usa el UUID proporcionado por la API

                # Separar datos en `jams` y listas anidadas
                main_record = {
                    k: v for k, v in record.items() if not isinstance(v, list)
                }
                nested_lists = {k: v for k, v in record.items() if isinstance(v, list)}

                # Insertar en la tabla principal `jams`
                jams_columns = ", ".join(
                    [
                        f"{self.db_columns_map[self.table_name][k]}"
                        for k in main_record.keys()
                    ]
                )
                jams_placeholders = ", ".join(["%s"] * len(main_record))
                sql_jams = (
                    f"INSERT INTO jams ({jams_columns}) VALUES ({jams_placeholders})"
                )
                cur.execute(sql_jams, tuple(main_record.values()))

                # Preparar inserciones en lote para `jams_segments` y `jams_lines`
                for attr, nested_list in nested_lists.items():
                    nested_table = f"{self.table_name}_{attr}"

                    # Convertir cada diccionario de la lista en tupla, agregando `jam_uuid`
                    batch_data = [
                        (
                            jam_uuid,
                            *tuple(nested_item.values()),
                            *tuple((p + 1,)),
                        )
                        for p, nested_item in enumerate(nested_list)
                    ]

                    # Extraer columnas de la primera entrada de la lista
                    nested_columns = ", ".join(
                        [f"{self.table_name}_uuid"]
                        + list(
                            [
                                self.db_columns_map[self.table_name][attr][k]
                                for k in nested_list[0].keys()
                            ]
                        )
                        + ["position"]
                    )
                    nested_placeholders = ", ".join(["%s"] * len(batch_data[0]))

                    # Inserción por lotes en la tabla correspondiente (`jams_segments` o `jams_lines`)
                    sql_nested = f"INSERT INTO {nested_table} ({nested_columns}) VALUES ({nested_placeholders})"
                    cur.executemany(sql_nested, batch_data)

        # Cerrar el cursor
        cur.close()

    @db_connection
    def update_endreport_to_db(self, endreport:int, uuid:str)->None:
        if self.table_name is None:
            return
        cur = None
        try:
            assert isinstance(self.db, connection), "Error in db connection"
            cur = self.db.cursor()
            cur.execute(
                "UPDATE "
                + self.table_name
                + " SET end_pub_millis = %s WHERE uuid = %s",
                (endreport, uuid),
            )

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")
        finally:
            if cur is not None:
                cur.close()


    @db_connection
    def update_endreports_to_db(self, from_new_data:bool=False)->int:
        if self.table_name is None:
            return 0

        cur = None
        try:
            assert isinstance(self.db, connection), "Error in db connection"
            cur = self.db.cursor()

            now = int((datetime.now(tz=pytz.utc).timestamp() - (5 * 60 / 2)) * 1000)

            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            not_ended = {d[0] for d in cur.fetchall()}

            if from_new_data:
                elements = [
                    (now, uuid) for uuid in not_ended if uuid not in self.index_map
                ]
            else:
                elements = [
                    (d["endreport"], d["uuid"])
                    for d in self.data
                    if d["uuid"] in not_ended
                    and "endreport" in d
                    and d["endreport"] is not None
                ]

            cur.executemany(
                "UPDATE "
                + self.table_name
                + " SET end_pub_millis = %s WHERE uuid = %s",
                elements,
            )

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")
        finally:
            if cur is not None:
                cur.close()

        return len(elements)

    @db_connection
    def update_endreports_to_db_from_file(self, filename=None):
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para leer el archivo")

        try:
            events = Events(filename=filename, table_name=self.table_name)
            changes = events.update_endreports_to_db()

            print(
                f"Se actualizaron {changes} elementos en {self.table_name} en end report"
            )

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")

