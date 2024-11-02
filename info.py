import time
from waze.events import Events
from waze.api import WazeAPI


def main():
    waze_api = WazeAPI()

    while True:
        print("Actualizando...")
        data = waze_api.get_data()
        try:
            if "alerts" in data:
                # 1. Capturar nuevas alertas
                alerts_api = Events(data["alerts"], table_name="alerts")
                # 2. Limpiar datos
                alerts_api.clean_data()
                # 3. Leer alertas no terminadas de la db
                alerts_not_ended_db = Events(table_name="alerts")
                alerts_not_ended_db.fetch_from_db(not_ended=True)
                # 4. Obtener nuevas alertas
                new_alerts = alerts_api - alerts_not_ended_db
                # 5. Insertar nuevas alertas en db
                new_alerts.insert_to_db()
                # 6. Actualizar en db las alertas terminadas
                new_alerts.update_endreports_to_db(from_new_data=True)
                print(f"Nuevas alertas: {len(new_alerts.data)}")

            if "jams" in data:
                # 1. Capturar nuevos eventos de congestión
                jams_api = Events(data["jams"], table_name="jams")
                # 2. Limpiar datos
                jams_api.clean_data()
                # 3. Leer eventos de congestión no terminados de la db
                jams_not_ended_db = Events(table_name="jams")
                jams_not_ended_db.fetch_from_db(not_ended=True)
                # 4. Obtener nuevos eventos de congestión
                new_jams = jams_api - jams_not_ended_db
                # 5. Insertar nuevos eventos de congestión en db
                new_jams.insert_to_db()
                # 6. Actualizar en db los eventos de congestión terminados
                new_jams.update_endreports_to_db(from_new_data=True)
                print(f"Nuevos eventos de congestión: {len(new_jams.data)}")

        except Exception as e:
            print(f"Error actualizando: {e}")

        print("Actualizado.")
        time.sleep(60 * 5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Saliendo...")
        exit(0)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
