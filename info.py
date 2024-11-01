import time
from waze.events import Events
from waze.api import WazeAPI


def main():
    waze_api = WazeAPI()
    alerts_data = Events(filename="data/alerts.json")
    jams_data = Events(filename="data/jams.json")

    while True:
        print("Actualizando...")
        data = waze_api.get_data()

        if "alerts" in data:
            alerts_api = Events(data["alerts"], table_name="alerts")
            alerts_api.clean_data()
        else:
            alerts_api = Events()

        if "jams" in data:
            jams_api = Events(data["jams"], table_name="jams")
            jams_api.clean_data()
        else:
            jams_api = Events()

        print(f"Nuevas alertas: {len(alerts_api.pending_endreports)}")
        print(f"Nuevos eventos de congestión: {len(jams_api.pending_endreports)}")

        alerts_data = Events(filename="data/alerts.json", table_name="alerts")
        jams_data = Events(filename="data/jams.json", table_name="jams")

        alerts_data += alerts_api
        jams_data += jams_api

        alerts_data.write_file()
        jams_data.write_file()

        print(f"Alertas: {len(alerts_data.data)}")
        print(f"Jams: {len(jams_data.data)}")

        try:
            alerts_not_ended = Events(table_name="alerts")
            jams_not_ended = Events(table_name="jams")

            # alerts_not_ended.update_endreports_to_db_from_file(
            #     filename="data/alerts.json"
            # )
            # jams_not_ended.update_endreports_to_db_from_file(filename="data/jams.json")

            alerts_not_ended.fetch_from_db(not_ended=True)
            jams_not_ended.fetch_from_db(not_ended=True)

            alerts_put = alerts_not_ended + alerts_api
            jams_put = jams_not_ended + jams_api

            alerts_put.insert_to_db()
            jams_put.insert_to_db()
        except Exception as e:
            print(f"Error actualizando: {e}")

        print("Actualizado.")
        time.sleep(60 * 5)


if __name__ == "__main__":
    main()
