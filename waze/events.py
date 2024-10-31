import json
from datetime import datetime
import pytz
import shutil
import os
from config import DATABASE_CONFIG
import psycopg2
from functools import wraps


def db_connection(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.table_name == "":
            raise ValueError("No se ha especificado la tabla de la base de datos")
        if self.db is None or self.db.closed:
            self.db = psycopg2.connect(**DATABASE_CONFIG)
            self.db.set_client_encoding("UTF8")

        try:
            func(self, *args, **kwargs)
        except psycopg2.Error as e:
            self.db.rollback()
            raise ValueError(f"Error executing database query: {e}")
        else:
            self.db.commit()
        finally:
            self.db.close()
            self.db = None

    return wrapper


class Events:
    def __init__(self, data=[], filename=None, table_name=""):
        self.data = data
        self.filename = filename
        self.table_name = table_name
        self.pending_endreports = set()
        self.db = None
        self.index_map = {}
        # Mapeo de columnas de Waze a columnas de la base de datos
        # Se usa para insertar datos en la base de datos
        # Cada tabla tiene diccionarios dentro, para esos casos se considera [nombre_atributo]_id
        # Además existe una tabla para cada diccionario dentro de los datos, con nombre [nombre_tabla]_[nombre_atributo]
        self.db_columns_map = {
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
                    "segment": "segment",
                    "ID": "segment_id",
                    "position": "position",
                    "fromNode": "from_node",
                    "toNode": "to_node",
                    "isForward": "is_forward",
                },
                "pubMillis": "pub_millis",
                "endreport": "end_pub_millis",
            },
        }

        if self.filename is not None and len(self.data) == 0:
            self.read_file()

        if len(self.data) > 0:
            self.update_index_map()

    def __end__(self):
        if self.db is not None and not self.db.closed:
            self.db.close()

    def __add__(self, other):
        if isinstance(other, Events):
            if len(other.data) > 0:
                new_data = [
                    d for d in other.data if d["uuid"] not in self.pending_endreports
                ]
            elif len(self.data) > 0:
                new_data = [
                    d for d in self.data if d["uuid"] not in other.pending_endreports
                ]

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
        else:
            raise TypeError("Operador + no soportado")

    def update_index_map(self):
        if len(self.data) > 0:
            self.index_map = {d["uuid"]: i for i, d in enumerate(self.data)}
            self.update_pending_endreports()
        else:
            self.index_map = {}

    def read_file(self, filename=None):
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para leer el archivo")

        if filename is not None:
            self.filename = filename

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
        return self.data

    def write_file(self, filename=None):
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para crear el archivo")

        if filename is not None:
            self.filename = filename

        if not os.path.exists(os.path.dirname(self.filename)):
            os.makedirs(os.path.dirname(self.filename))

        if os.path.exists(self.filename):
            shutil.copyfile(self.filename, self.filename + ".bak")

        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def update_pending_endreports(self):
        self.pending_endreports = {
            d["uuid"]
            for d in self.data
            if "endreport" not in d or d["endreport"] is None
        }

    def end_report(self, uuid):
        now = int((datetime.now(tz=pytz.utc).timestamp() - (5 * 60 / 2)) * 1000)

        idx = self.index_map.get(uuid, None)

        if idx is not None:
            self.data[idx]["endreport"] = now
            self.update_endreport_to_db(now, uuid)
            self.pending_endreports.discard(uuid)

    def clean_data(self):
        deletions = set(k for d in self.data for k in d) - set(
            self.db_columns_map[self.table_name].keys()
        )
        for item in self.data:
            for key in deletions:
                if key in item:
                    item.pop(key, None)

    def format_data(self):
        if not isinstance(self.data, list):
            return
        if len(self.data) > 0:
            cols = [
                k
                for k in self.db_columns_map[self.table_name].keys()
                if k not in ["line", "segments"]
            ]
            self.data = [{k: v for k, v in zip(cols, d)} for d in self.data]

    @db_connection
    def fetch_from_db(self, not_ended=False):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        cur = self.db.cursor()
        if not_ended:
            cur.execute(
                "SELECT * FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
        else:
            cur.execute("SELECT * FROM " + self.table_name)

        self.data = cur.fetchall()

        self.format_data()
        self.update_index_map()

    @db_connection
    def get_all_from_db(self, not_ended=True):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        if not_ended:
            cur = self.db.cursor()
            cur.execute(
                "SELECT * FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            events = cur.fetchall()
            return events

        cur = self.db.cursor()
        cur.execute("SELECT * FROM " + self.table_name)
        events = cur.fetchall()
        return events

    @db_connection
    def insert_to_db(self):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        cur = self.db.cursor()

        try:
            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            not_ended = {d[0] for d in cur.fetchall()}
        except psycopg2.Error as e:
            raise ValueError(f"Error checking existing data: {e}")

        # Lista de uuids que no se encuentran en la base de datos
        data = [self.data[i] for d, i in self.index_map.items() if d not in not_ended]

        for record in data:
            # Identificar si estamos en el caso de `alerts` o `jams`
            if self.table_name == "alerts":
                # Caso `alerts`: Crear referencia `location_id` para `alerts_location`
                location_data = record.pop(
                    "location", None
                )  # Extraer el diccionario de `location`
                if location_data:
                    # Insertar datos en `alerts_location` y obtener el `location_id`
                    sql_location = "INSERT INTO alerts_location (x, y, segment) VALUES (%s, %s, %s) RETURNING id"
                    cur.execute(
                        sql_location,
                        (
                            location_data["x"],
                            location_data["y"],
                            location_data["segment"]
                            if "segment" in location_data
                            else 0,
                        ),
                    )
                    location_id = cur.fetchone()[0]  # Obtener el `location_id` generado

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
                        (jam_uuid, *tuple(nested_item.values()))
                        for nested_item in nested_list
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
                    )
                    nested_placeholders = ", ".join(["%s"] * len(batch_data[0]))

                    # Inserción por lotes en la tabla correspondiente (`jams_segments` o `jams_lines`)
                    sql_nested = f"INSERT INTO {nested_table} ({nested_columns}) VALUES ({nested_placeholders})"
                    cur.executemany(sql_nested, batch_data)

            # Confirmar la transacción después de cada `record`
            self.db.commit()

        # Cerrar el cursor
        cur.close()
        return True

    @db_connection
    def update_endreport_to_db(self, endreport, uuid):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        try:
            cur = self.db.cursor()
            cur.execute(
                "UPDATE "
                + self.table_name
                + " SET end_pub_millis = %s WHERE uuid = %s",
                (endreport, uuid),
            )
            self.db.commit()

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")
        finally:
            cur.close()

        return True

    @db_connection
    def update_endreports_to_db(self):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        try:
            cur = self.db.cursor()

            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            not_ended = {d[0] for d in cur.fetchall()}

            cur.executemany(
                "UPDATE "
                + self.table_name
                + " SET end_pub_millis = %s WHERE uuid = %s",
                [
                    (d["endreport"] if "endreport" in d else None, d["uuid"])
                    for d in self.data
                    if d["uuid"] in not_ended
                ],
            )
            self.db.commit()

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")
        finally:
            cur.close()

    @db_connection
    def update_endreports_to_db_from_file(self, filename=None):
        if self.db is None:
            raise ValueError("No se ha conectado a la base de datos")

        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para leer el archivo")

        try:
            cur = self.db.cursor()

            cur.execute(
                "SELECT uuid FROM " + self.table_name + " WHERE end_pub_millis IS NULL"
            )
            not_ended = {d[0] for d in cur.fetchall()}

            events = Events(filename=filename, table_name=self.table_name)
            to_db = Events(table_name=self.table_name)

            changes = 0

            for d in events.data:
                if d["uuid"] in not_ended:
                    to_db.data.append(d)

                    changes += 1

            print(
                f"Se actualizaron {changes} elementos en {self.table_name} en end report"
            )
            self.db.commit()

        except psycopg2.Error as e:
            raise ValueError(f"Error updating data to database: {e}")
        finally:
            cur.close()

        if changes > 0:
            to_db.update_endreports_to_db()
