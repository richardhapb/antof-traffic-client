import json
from datetime import datetime
import pytz


class Events:
    def __init__(self, data=[], filename=None):
        self.data = data
        self.filename = filename
        self.pending_endreports = {}

        if filename is not None:
            self.read_file()

        self.update_pending_endreports()

    def __add__(self, other):
        if isinstance(other, Events):
            last_24h = int((datetime.now().timestamp() - (24 * 60 * 60)) * 1000)
            existing_uuids = {d["uuid"] for d in self.data if d["pubMillis"] > last_24h}

            new_data = [d for d in other.data if d["uuid"] not in existing_uuids]

            events = Events(new_data, self.filename)
            events.pending_endreports = (
                self.pending_endreports | other.pending_endreports
            )

            # Update pending endreports if an uuid is not found in the new data
            for uuid in existing_uuids:
                if uuid not in other.pending_endreports:
                    events.end_report(uuid)

            return events
        else:
            raise TypeError("Operador + no soportado")

    def read_file(self, filename=None):
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para leer el archivo")

        if filename is not None:
            self.filename = filename

        with open(self.filename, "r") as f:
            self.data = json.load(f)
        return self.data

    def write_file(self, filename=None):
        if filename is None and self.filename is None:
            raise ValueError("Se requiere una ruta para crear el archivo")

        if filename is not None:
            self.filename = filename

        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def update_pending_endreports(self):
        self.pending_endreports = {d["uuid"] for d in self.data if "endreport" not in d}

    def end_report(self, uuid):
        tz = pytz.utc
        now = int((datetime.now(tz).timestamp() - (5 * 60 / 2)) * 1000)

        idx = next((i for i, d in enumerate(self.data) if d["uuid"] == uuid), None)

        if idx is not None:
            self.data[idx]["endreport"] = now
            self.pending_endreports.discard(uuid)
            return True
        else:
            return False
