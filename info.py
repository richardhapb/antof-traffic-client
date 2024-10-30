import time
from waze.events import Events
from waze.api import WazeAPI


def main():
    waze_api = WazeAPI()
    alerts_data = Events(filename="data/alerts.json")
    jams_data = Events(filename="data/jams.json")

    while True:
        init_time = time.perf_counter()
        print("Actualizando...")
        data = waze_api.get_data()

        if "alerts" in data:
            alerts_api = Events(data["alerts"])
        else:
            alerts_api = Events()

        if "jams" in data:
            jams_api = Events(data["jams"])
        else:
            jams_api = Events()

        print(f"Nuevas alertas: {len(alerts_api.pending_endreports)}")
        print(f"Nuevos eventos de congestión: {len(jams_api.pending_endreports)}")

        alerts_data += alerts_api
        jams_data += jams_api

        alerts_data.write_file()
        jams_data.write_file()

        print("Actualizado.")
        print(f"Tiempo: {time.perf_counter() - init_time:.2f} segundos")
        time.sleep(60 * 5)


if __name__ == "__main__":
    main()
