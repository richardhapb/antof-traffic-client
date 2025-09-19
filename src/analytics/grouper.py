from typing import Any, TYPE_CHECKING, ClassVar

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# Antofagasta's perimeter
XMIN = -70.43627
XMAX = -70.36259

YMIN = -23.724215
YMAX = -23.485813

if TYPE_CHECKING:
    from .alerts import Alerts


class GroupNotFoundError(Exception):
    """Exception throwed when group is not found"""

    def __init__(self, point: tuple[int, int]):
        super().__init__(f"The point {point} is not in any quadrant")


class Grouper:
    """Class for group alerts according to geospatial location (x, y)"""

    concepts: ClassVar[dict[str, str]] = {
        "ACCIDENT": "Accidentes",
        "JAM": "Congestión",
        "HAZARD": "Peligros",
        "ROAD_CLOSED": "Cierres de vías",
        "ALL": "Eventos",
    }

    def __init__(self, data: gpd.GeoDataFrame, grid_dim: tuple[int, int]) -> None:
        self.grid: tuple[np.ndarray[Any, Any], ...] | None = None
        self.type: str | None = None
        self.x_grid: np.ndarray | None = None
        self.y_grid: np.ndarray | None = None

        self.get_grid(data, *grid_dim)

    def get_grid(
        self, data: gpd.GeoDataFrame, n_x_div: int, n_y_div: int
    ) -> tuple[np.ndarray[Any, Any], ...]:
        """
        Calculate and set a grid, a two-dimensional array that represents the segments of the map

        Args:
            data: `GeoDataFrame` containing the geo-data
            n_x_div: number of horizontal segment divisions
            n_y_div: number of vertical segment divisions

        Returns:
            The calculated two-dimensional array

        """
        geometry = data.geometry
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
    ) -> int:
        """
        Get the consecutive quadrant number from the grid

        for example:

              +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
              |     |     |     |     |     |     |     |     |     |     |
              | 11  | 12  |[13] | 14  | 15  | 16  | 17  | 18  | 19  | 20  |
              |     |     |     |     |     |     |     |     |     |     |
              +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
              |     |     |     |     |     |     |     |     |     |     |
              |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  |
              |     |     |     |     |     |     |     |     |     |     |
              +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

             13 is selected for x_pos=2 and y_pos=1

        """
        return self.y_len * x_pos + y_pos + 1

    def get_quadrant(self, point: tuple[int, int]) -> tuple[int, int]:
        """
        Get the indexes for longitude and latitude

        Args:
            point: (x, y) coordinates

        Returns:
            A tuple containing the indexes (x, y) where the quadrant is located, starting
            in the lower-left corner

        Example:
              +-----+-----+-22--+-----+-----+-----+-----+-----+-----+-----+
              |     |     |     |     |     |     |     |     |     |     |
              | 11  | 12  |[13] | 14  | 15  | 16  | 17  | 18  | 19  | 20  |
              |     |    72    73     |     |     |     |     |     |     |
              +-----+-----+-23--+-----+-----+-----+-----+-----+-----+-----+
              |     |     |     |     |     |     |     |     |     |     |
              |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  |
              |     |     |     |     |     |     |     |     |     |     |
              +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

             Returns 13 if the point is (72.5, 22.5)

        """
        x_pos, y_pos = -1, -1

        if self.x_grid is None or self.y_grid is None:
            return (-1, -1)

        for xi in range(len(self.x_grid[0])):
            if (
                xi < len(self.x_grid[0]) - 1
                and point[0] >= self.x_grid[0][xi]
                and point[0] <= self.x_grid[0][xi + 1]
            ):
                x_pos = xi
                break

        for yi in range(len(self.y_grid)):
            if (
                yi < len(self.y_grid) - 1
                and point[1] >= self.y_grid[yi][0]
                and point[1] <= self.y_grid[yi + 1][0]
            ):
                y_pos = yi
                break

        if x_pos < 0 or y_pos < 0:
            raise GroupNotFoundError(point)

        return x_pos, y_pos

    def get_center_points(self) -> tuple:
        """
        Get the center points of the quadrants

        Returns:
            An two-dimensional array with the center points

        """

        if self.grid is None:
            return (-1, -1)
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

    """
    Plotting helpers for exploratory analysis

    This code is for exploratory purposes and contains a lot of repetitive code
    but it is not used; I need to review it for refactoring or removal
    """

    def plot_qty_day(self, alerts: "Alerts") -> Figure:
        fig, ax = plt.subplots()
        grouped_day = alerts.group_by_day()

        fig.set_size_inches((4.5, 9.5))
        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        labels = [False, False, False]
        for i, xp in enumerate(xc[0]):
            for j, yp in enumerate(yc.T[0]):
                quad = self.calc_quadrant(i, j)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                group_freq = (
                    grouped_day[grouped_day["group"] == quad]["qty/day"].astype(np.float16)[0]
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
                    round(group_freq, 1),  # type: ignore
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

        assert alerts.data.crs is not None, "CRS is None"

        cx.add_basemap(
            ax,
            crs=alerts.data.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore
        )

        ax.set_title(f"{self.concepts[self.type]} por día por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.legend(fontsize=8, loc="upper left")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1)

        return fig

    def plot_qty_day_alpha(self, data: gpd.GeoDataFrame, with_numbers=False) -> Figure:
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
                    ax.text(xp - 150, yp - 150, str(quad), fontsize=6, alpha=0.5)

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=(
                        (
                            data["group"].value_counts()[
                                quad
                                if quad in data["group"].value_counts().index
                                else data["group"].value_counts().idxmin()
                            ]
                            - data["group"].value_counts().min()
                        )
                        / (data["group"].value_counts().max() - data["group"].value_counts().min())
                    ),
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax,
            crs=data.crs.to_string() if data.crs else '',
            source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    def plot_qty(self, data: gpd.GeoDataFrame):
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
                    data.value_counts("group")
                    .reset_index()[data.value_counts("group").reset_index()["group"] == quad][
                        "count"
                    ]
                    .astype(np.int16)[0]
                    if quad in data.value_counts("group").reset_index()["group"].values
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
                    str(qty),
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
            ax,
            crs=data.crs.to_string() if data.crs else '',
            source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.legend(fontsize=8, loc="upper left")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.87, bottom=0.1)

        return fig

    def plot_qty_alpha(self, data: gpd.GeoDataFrame, with_numbers: bool = False) -> Figure:
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
                    ax.text(xp - 150, yp - 150, str(quad), fontsize=6, alpha=0.5)

                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=(
                        (
                            data["group"].value_counts()[
                                quad
                                if quad in data["group"].value_counts().index
                                else data["group"].value_counts().idxmin()
                            ]
                            - data["group"].value_counts().min()
                        )
                        / (data["group"].value_counts().max() - data["group"].value_counts().min())
                    ),
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax,
            crs=data.crs.to_string() if data.crs else '',
            source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore
        )

        ax.set_title(f"{self.concepts[self.type]} por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    def plot_with_numbers(self, data: gpd.GeoDataFrame) -> Figure:
        fig, ax = plt.subplots()
        fig.set_size_inches((4.5, 9.5))

        xc, yc = self.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = self.calc_quadrant(i, j)
                ax.text(xp - 250, yp - 150, str(quad), fontsize=6, alpha=0.5)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                alpha = (
                    data["group"].value_counts()[
                        quad
                        if quad in data["group"].value_counts().index
                        else data["group"].value_counts().idxmin()
                    ]
                    - data["group"].value_counts().min()
                ) / (data["group"].value_counts().max() - data["group"].value_counts().min())

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
            ax,
            crs=data.crs.to_string() if data.crs else '',
            source=cx.providers.OpenStreetMap.Mapnik,  # type: ignore
        )

        ax.set_title("Accidentes por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")

        return fig
