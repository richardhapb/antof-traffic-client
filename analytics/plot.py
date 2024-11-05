import numpy as np
import matplotlib.pyplot as plt
from utils import utils
import geopandas as gpd
import contextily as cx


class Plot:
    def heat_map(self, data: gpd.GeoDataFrame):
        event = data.copy()

        if len(event) == 0:
            return None
        event = event[["geometry"]]
        event = event.to_crs(epsg=3857)  # Para visualización correcta en plano

        fig, ax = plt.subplots()
        fig.set_size_inches((4.5, 9.5))
        sc = event["geometry"].plot(ax=ax, alpha=0.5, cmap="GnBu")
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
        # Añadir la barra de colores
        cbar = sc.get_figure().colorbar(sc.collections[0], ax=ax)
        cbar.set_label("Frecuencia")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        plt.xticks(rotation=45)
        fig.suptitle("Eventos en\nAntofagasta", fontweight="bold")
        fig.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
        fig.tight_layout()

        return fig

    def hourly_report(self, data: gpd.GeoDataFrame):
        """
        Reportes de eventos por día de la semana
        """

        if data.empty:
            return None

        hourly_reports = utils.hourly_group(data)

        # Graficar la fluctuación de reportes por hora diferenciando entre semana y fin de semana
        fig, ax = plt.subplots()
        fig.set_size_inches((10, 6))

        # Ajustar las posiciones de las barras
        bar_width = 0.4
        r1 = np.arange(len(hourly_reports))
        r2 = [x + bar_width for x in r1]

        if "s" in hourly_reports.columns:
            plt.bar(
                r1,
                hourly_reports.loc[:, "s"],
                color="salmon",
                width=bar_width,
                edgecolor="grey",
                label="Semana",
            )
        if "f" in hourly_reports.columns:
            plt.bar(
                r2,
                hourly_reports.loc[:, "f"],
                color="skyblue",
                width=bar_width,
                edgecolor="grey",
                label="Fin de Semana / Feriado",
            )

        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Número de reportes")

        ax.set_xticks(
            [r + bar_width / 2 for r in range(len(hourly_reports))],
            hourly_reports.index,
        )
        plt.grid(True, alpha=0.3)
        fig.suptitle(
            "Eventos durante el día\n(Semana vs. Fin de Semana)",
            fontweight="bold",
        )
        ax.legend(loc="upper left")
        fig.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
        fig.tight_layout()

        return fig

    def daily_report(self, data: gpd.GeoDataFrame):
        """
        Reportes de eventos por día de la semana
        """

        if data.empty:
            return None

        daily_reports = utils.daily_group(data)

        fig, ax = plt.subplots()
        fig.set_size_inches((10, 6))

        # Ajustar las posiciones de las barras
        bar_width = 0.4
        r1 = np.arange(len(daily_reports))
        r2 = [x + bar_width for x in r1]

        if "s" in daily_reports.columns:
            plt.bar(
                r1,
                daily_reports.loc[:, "s"],
                color="salmon",
                width=bar_width,
                edgecolor="grey",
                label="Semana",
            )
        if "f" in daily_reports.columns:
            plt.bar(
                r2,
                daily_reports.loc[:, "f"],
                color="skyblue",
                width=bar_width,
                edgecolor="grey",
                label="Fin de Semana / Feriado",
            )

        ax.set_xlabel("Día de la semana")
        ax.set_ylabel("Número de reportes")

        ax.set_xticks(
            [r + bar_width / 2 for r in range(len(daily_reports))], daily_reports.index
        )
        plt.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.suptitle(
            "Eventos durante el mes\n(Semana vs. Fin de Semana)",
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.15, right=0.8, top=0.9, bottom=0.1)
        fig.tight_layout()

        return fig
