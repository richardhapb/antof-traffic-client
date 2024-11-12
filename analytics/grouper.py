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
        "ROAD_CLOSED": "Cierres de vías",
        "ALL": "Eventos",
    }

    def __init__(self, data: gpd.GeoDataFrame | pd.DataFrame):
        self.data = data.copy()
        self.grid = None
        self.type = None
        self.x_grid = None
        self.y_grid = None
        pass

    def get_grid(self, n_x_div: int, n_y_div: int):
        geometry = self.data.geometry
        bounds_x = np.array(
            np.linspace(
                geometry.to_crs(epsg=3857).geometry.x.min(),
                geometry.to_crs(epsg=3857).geometry.x.max(),
                n_x_div,
            )
        )
        bounds_y = np.array(
            np.linspace(
                geometry.to_crs(epsg=3857).geometry.y.min(),
                geometry.to_crs(epsg=3857).geometry.y.max(),
                n_y_div,
            )
        )

        self.grid = np.meshgrid(bounds_x, bounds_y)
        self.x_grid = self.grid[0]
        self.y_grid = self.grid[1]
        self.x_len = self.x_grid.shape[1] - 1
        self.y_len = self.y_grid.shape[0] - 1
        return self.grid

    def calc_quadrant(
        self,
        x_pos: int,
        y_pos: int,
    ):
        return self.y_len * x_pos + y_pos + 1

    def get_quadrant(self, point: tuple):
        """
        point: (x, y) positions, beginning in lower left corner
        """
        x_pos, y_pos = -1, -1

        for xi in range(len(self.x_grid[0])):
            if (
                xi < len(self.x_grid[0]) - 1
                and point[0] >= self.x_grid[0][xi]
                and point[0] <= self.x_grid[0][xi + 1]
            ):
                x_pos = xi

        for yi in range(len(self.y_grid)):
            if (
                yi < len(self.y_grid) - 1
                and point[1] >= self.y_grid[yi][0]
                and point[1] <= self.y_grid[yi + 1][0]
            ):
                y_pos = yi

        if x_pos < 0 or y_pos < 0:
            raise ValueError(f"El punto {point} no se encuentra en ningún cuadrante")

        quadrant = x_pos, y_pos

        return quadrant

    def get_center_points(self):
        # X
        center_points_x = np.zeros((self.y_len, self.x_len))

        x_half = (self.grid[0][0][1] - self.grid[0][0][0]) / 2

        for x in range(len(self.grid[0][0]) - 1):
            center_points_x[0][x] = self.grid[0][0][x] + x_half

        center_points_x[:][:] = center_points_x[0][:]

        # Y
        center_points_y = np.zeros((self.y_len, self.x_len))

        y_half = (self.grid[1][1][0] - self.grid[1][0][0]) / 2

        for y in range(len(self.grid[1]) - 1):
            center_points_y[y][0] = self.grid[1][y][0] + y_half

        for c in range(len(center_points_y)):
            center_points_y[c][:] = center_points_y[c][0]

        return center_points_x, center_points_y

    def group(
        self,
        grid_dim: tuple,
        concepts: list = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"],
    ):
        grouped = self.data.copy()
        self.get_grid(*grid_dim)

        grouped["group"] = grouped.to_crs(epsg=3857).geometry.apply(
            lambda x: self.calc_quadrant(
                *self.get_quadrant(
                    (
                        x.x,
                        x.y,
                    ),
                ),
            ),
        )

        self.data = grouped

        if len(concepts) == 1:
            self.type = concepts[0]
        else:
            self.type = "ALL"

        return self

    def filter_by_group_time(self, timedelta_min: int, inplace: bool = False):
        if self.data is None or "pubMillis" not in self.data.columns:
            return None
        events2 = self.data.copy()

        # Asegurar que 'pubMillis' es np.int64 y está en milisegundos
        if not isinstance(events2["pubMillis"].iloc[0], np.int64):
            events2["pubMillis"] = (
                (pd.to_numeric(events2["pubMillis"]) / 1_000_000)
                .round()
                .astype(np.int64)
            )

        step = np.int64(60_000 * timedelta_min)  # step en milisegundos

        # Calcular intervalos ajustando 'pubMillis' al múltiplo más cercano de 'step'
        events2["interval"] = ((events2["pubMillis"]) // step) * step

        min_t = (events2["pubMillis"].min() // step) * step
        max_t = (events2["pubMillis"].max() // step) * step

        intervals = np.arange(min_t, max_t + step, step).astype(np.int64)
        groups = events2["group"].astype(str).unique()
        types = events2["type"].astype(str).unique()

        combinations = pd.MultiIndex.from_product(
            [intervals, groups, types], names=["interval", "group", "type"]
        ).to_frame(index=False)

        # Asegurar tipos consistentes
        events2["group"] = events2["group"].astype(str)
        events2["type"] = events2["type"].astype(str)

        merged = pd.merge(
            combinations, events2, how="left", on=["interval", "group", "type"]
        )

        # Si deseas eliminar filas solo donde ciertas columnas son nulas
        merged = merged.dropna()

        # Actualizar 'pubMillis' y convertir a datetime si es necesario
        merged["pubMillis"] = merged["interval"]
        merged = merged.drop(columns=["interval"])
        merged["group"] = merged["group"].astype(np.int16)

        if not isinstance(self.data["pubMillis"].iloc[0], np.int64):
            merged["pubMillis"] = pd.to_datetime(
                (merged["pubMillis"] * 1_000_000), unit="ns"
            )
            merged["pubMillis"] = merged["pubMillis"].dt.tz_localize("America/Santiago")
        else:
            merged["pubMillis"] = pd.to_datetime(merged["pubMillis"], unit="ms")
            merged["pubMillis"] = merged["pubMillis"].dt.tz_localize("America/Santiago")

        result = merged

        if inplace:
            self.data = result

        return result

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
        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        labels = [False, False, False]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)
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

        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)
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

        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        labels = [False, False, False]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)

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

        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)
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

    def plot_with_numbers(self):
        fig, ax = plt.subplots()
        fig.set_size_inches((4.5, 9.5))

        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)
                ax.text(xp - 250, yp - 150, quad, fontsize=6, alpha=0.5)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                alpha = (
                    self.data["group"].value_counts()[
                        quad
                        if quad in self.data["group"].value_counts().index
                        else self.data["group"].value_counts().idxmin()
                    ]
                    - self.data["group"].value_counts().min()
                ) / (
                    self.data["group"].value_counts().max()
                    - self.data["group"].value_counts().min()
                )

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=alpha if alpha > 0.05 else 0.1,
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax, crs=self.data.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
        )

        ax.set_title("Accidentes por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")

        return fig
