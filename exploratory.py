from analytics.plot import Plot
from utils import utils
import matplotlib.pyplot as plt
import datetime
import pytz


def main():
    tz = "America/Santiago"

    since = int(
        (
            datetime.datetime.now(pytz.timezone(tz)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            - datetime.timedelta(days=1)
        ).timestamp()
        * 1000
    )

    alerts = utils.load_data("alerts", mode="since", epoch=since)
    plot = Plot()

    # Graficar el mapa de eventos
    fig = plot.heat_map(alerts.to_gdf())
    fig.suptitle(
        f"Eventos en Antofagasta\ndesde {datetime.datetime.fromtimestamp(since / 1000).strftime('%d-%m-%Y')}",
        fontweight="bold",
    )
    fig.savefig("graph/alerts_heat_map.png")

    events = utils.extract_event(alerts.to_gdf(tz=tz), ["ACCIDENT"])

    fig = plot.heat_map(events)
    fig.suptitle(
        f"Accidentes en Antofagasta\ndesde {datetime.datetime.fromtimestamp(since / 1000).strftime('%d-%m-%Y')}",
        fontweight="bold",
    )
    fig.savefig("graph/alerts_heat_map_accidents.png")

    fig = plot.hourly_report(events)
    fig.suptitle(
        f"Accidentes cada hora desde {datetime.datetime.fromtimestamp(since / 1000).strftime('%d-%m-%Y')}",
        fontweight="bold",
    )
    fig.savefig("graph/alerts_hourly_report.png")

    fig = plot.daily_report(events)
    fig.suptitle(
        f"Accidentes diarios desde {datetime.datetime.fromtimestamp(since / 1000).strftime('%d-%m-%Y')}",
        fontweight="bold",
    )
    fig.savefig("graph/alerts_daily_report.png")

    plt.show()


if __name__ == "__main__":
    main()
