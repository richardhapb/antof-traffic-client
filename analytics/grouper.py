from utils import utils
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as cx
import numpy as np


class Grouper:
    concepts = {
        "ACCIDENT": "Accidentes",
        "JAM": "Congestión",
        "HAZARD": "Peligros",
    }

    def __init__(self):
        self.data = None
        self.grid = None
        self.type = None
        pass

    def group(self, data: gpd.GeoDataFrame, grid_dim: tuple, concept):
        grouped = data.copy()
        grid = utils.grid(data, *grid_dim)
        self.grid = grid

        grouped["group"] = grouped.to_crs(epsg=3857).geometry.apply(
            lambda x: utils.calc_quadrant(
                *utils.get_quadrant(
                    *grid,
                    (
                        x.x,
                        x.y,
                    ),
                ),
                grid[0].shape[1] - 1,
            ),
        )

        self.data = grouped
        self.type = concept

        return self

    def group_by_day(self):
        grouped_day = (
            pd.DataFrame(
                {
                    "group": self.data.group.value_counts().keys(),
                    "qty/day": self.data.group.value_counts().values
                    / (self.data["inicio"].max() - self.data["inicio"].min()).days,
                }
            )
        ).sort_values(ascending=False, by="qty/day")

        return grouped_day

    def plot_qty_day(self):
        fig, ax = plt.subplots()
        grouped_day = self.group_by_day()

        fig.set_size_inches((4.5, 9.5))
        xc, yc = utils.get_center_points(self.grid)
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        labels = [False, False, False]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = utils.calc_quadrant(i, j, self.grid[0].shape[1] - 1)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                group_freq = (
                    np.float16(grouped_day[grouped_day["group"] == quad]["qty/day"])[0]
                    if quad in grouped_day["group"].values
                    else 0
                )
                color = "r" if group_freq > 0 else "g"
                color = "b" if group_freq > 0.5 else color
                label = ""
                if group_freq == 0 and not labels[0]:
                    label = f"Sin {self.concepts[self.type]}"
                    labels[0] = True

                if group_freq > 0 and not labels[1] and label == "":
                    label = f"{self.concepts[self.type]} por día"
                    labels[1] = True

                if group_freq > 0.5 and not labels[2] and label == "":
                    label = "Zona crítica"
                    labels[2] = True

                ax.text(
                    xp - 220,
                    yp - 150,
                    round(group_freq, 1),
                    fontsize=7,
                    alpha=0.8,
                )
                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=0.5,
                    color=color,
                    label=label,
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax,
            crs=self.data.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,
        )

        ax.set_title(f"{self.concepts[self.type]} por día por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.legend(fontsize=8, loc="upper left")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1)

        return fig

    def plot_qty_day_alpha(self, with_numbers=False):
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))

        xc, yc = utils.get_center_points(self.grid)
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = utils.calc_quadrant(i, j, self.grid[0].shape[1] - 1)
                xf = xp - between_x / 2
                yf = yp - between_y / 2

                if with_numbers:
                    ax.text(xp - 150, yp - 150, quad, fontsize=6, alpha=0.5)

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=(
                        (
                            self.data["group"].value_counts()[
                                quad
                                if quad in self.data["group"].value_counts().index
                                else self.data["group"].value_counts().idxmin()
                            ]
                            - self.data["group"].value_counts().min()
                        )
                        / (
                            self.data["group"].value_counts().max()
                            - self.data["group"].value_counts().min()
                        )
                    ),
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax, crs=self.data.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    def plot_qty(self):
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))

        xc, yc = utils.get_center_points(self.grid)
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        labels = [False, False, False]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = utils.calc_quadrant(i, j, self.grid[0].shape[1] - 1)

                qty = (
                    np.int16(
                        self.data.value_counts("group").reset_index()[
                            self.data.value_counts("group").reset_index()["group"]
                            == quad
                        ]["count"]
                    )[0]
                    if quad
                    in self.data.value_counts("group").reset_index()["group"].values
                    else 0
                )

                color = "r" if qty > 0 else "g"
                color = "b" if qty > 60 else color
                label = ""
                if qty == 0 and not labels[0]:
                    label = f"Sin {self.concepts[self.type]}"
                    labels[0] = True

                if qty > 0 and not labels[1] and label == "":
                    label = f"{self.concepts[self.type]} totales en el periodo"
                    labels[1] = True

                if qty > 60 and not labels[2] and label == "":
                    label = "Zona crítica"
                    labels[2] = True

                ax.text(
                    xp - 150,
                    yp - 150,
                    qty,
                    fontsize=6,
                    alpha=0.5,
                )
                xf = xp - between_x / 2
                yf = yp - between_y / 2

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=0.5,
                    color=color,
                    label=label,
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax, crs=self.data.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.legend(fontsize=8, loc="upper left")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1)

        return fig

    def plot_qty_alpha(self, with_numbers=False):
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))

        xc, yc = utils.get_center_points(self.grid)
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = utils.calc_quadrant(i, j, self.grid[0].shape[1] - 1)
                # ax.text(xp - 150, yp - 150, quad, fontsize=6, alpha=0.5)
                xf = xp - between_x / 2
                yf = yp - between_y / 2

                if with_numbers:
                    ax.text(xp - 150, yp - 150, quad, fontsize=6, alpha=0.5)

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=(
                        (
                            self.data["group"].value_counts()[
                                quad
                                if quad in self.data["group"].value_counts().index
                                else self.data["group"].value_counts().idxmin()
                            ]
                            - self.data["group"].value_counts().min()
                        )
                        / (
                            self.data["group"].value_counts().max()
                            - self.data["group"].value_counts().min()
                        )
                    ),
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax, crs=self.data.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig
