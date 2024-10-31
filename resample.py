# from aws import dynamodb
import json
import pytz
from datetime import datetime

with open("data/waze.json", "r") as f:
    data = json.load(f)

alerts = data["alerts"]
jams = data["jams"]

# dynamodb.put_batch("waze_alert_antof", alerts)

# Pasos subida a DynamoDB
# 1. Eliminar columnas innecesarias provenientes de Waze y modificar endreport a UTC
#   ALERTS
#       country
#       city
#       reportRating
#       reportByMunicipalityUser
#       confidence
#   JAMS
#       country
#       city
#       turnType
#       speed
#       id

for i in range(len(alerts)):
    if "endreport" not in alerts[i].keys():
        continue
    new_time = datetime.fromtimestamp(
        alerts[i]["endreport"] / 1000, pytz.timezone("America/Santiago")
    )
    alerts[i]["endreport"] = int(new_time.astimezone(pytz.utc).timestamp() * 1000)

for i in range(len(jams)):
    if "endreport" not in jams[i].keys():
        continue
    new_time = datetime.fromtimestamp(
        jams[i]["endreport"] / 1000, pytz.timezone("America/Santiago")
    )
    jams[i]["endreport"] = int(new_time.astimezone(pytz.utc).timestamp() * 1000)


alerts_deletions = [
    "country",
    "city",
    "reportRating",
    "reportByMunicipalityUser",
    "confidence",
    "reportDescription",
]
jams_deletions = [
    "country",
    "city",
    "turnType",
    "speed",
    "id",
    "blockingAlertUuid",
    "startNode",
]

for i in range(len(alerts)):
    for d in alerts_deletions:
        if d in alerts[i].keys():
            del alerts[i][d]

for i in range(len(jams)):
    for d in jams_deletions:
        if d in jams[i].keys():
            del jams[i][d]

with open("data/alerts.json", "w") as f:
    json.dump(alerts, f)

with open("data/jams.json", "w") as f:
    json.dump(jams, f)

# 2. Subir la base de datos completa a DynamoDB
# 3. Generar workflows para actualizar la base de datos a intervalos regulares, evitando duplicados
# 4. Generar un pipeline que realice los pasos anteriores

print(f"Hay {len(alerts)} alertas y {len(jams)} jams")
