import datetime

import matplotlib.pyplot as plt
import pytz

from analytics.plot import Plot
from utils import utils
from utils.utils import TZ


def main():

    since = int(
        (
            datetime.datetime.now(pytz.timezone(TZ)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            - datetime.timedelta(days=1)
        ).timestamp()
        * 1000
    )

    alerts = utils.get_data(since)
    plot = Plot()

    # Graficar el mapa de eventos
    fig = plot.heat_map(alerts.data)
    fig.suptitle(
        f"Eventos en Antofagasta\ndesde {datetime.datetime.fromtimestamp(since / 1000).strftime('%d-%m-%Y')}",
        fontweight="bold",
    )
    fig.savefig("graph/alerts_heat_map.png")

    events = alerts.data[alerts.data['type'] == "ACCIDENT"]

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
