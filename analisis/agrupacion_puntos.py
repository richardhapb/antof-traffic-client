# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %cd ..

# %%
from utils import utils
import matplotlib.pyplot as plt
import pandas as pd
import contextily as cx
import numpy as np

data = pd.DataFrame(utils.load_data("alerts").data)

# %%
alerts = utils.extract_event(data, ['ACCIDENT']).drop("uuid", axis=1)

# %%
antof_grid = utils.grid(alerts, 10, 20)
antof_grid

# %%
alerts_grouped = alerts.copy()
alerts_grouped["group"] = alerts_grouped.to_crs(epsg=3857).geometry.apply(
    lambda x: utils.calc_quadrant(
        *utils.get_quadrant(
            *antof_grid,
            (
                x.x,
                x.y,
            ),
        ),
        antof_grid[0].shape[1] - 1,
    ),
)   

# %%
alerts_grouped["group"].value_counts()

# %%
## Accidentes por grupo al día
grouped_day = (pd.DataFrame({"group": alerts_grouped.group.value_counts().keys(), "qty/day": alerts_grouped.group.value_counts().values / (alerts_grouped["inicio"].max() - alerts_grouped["inicio"].min()).days})).sort_values(ascending=False, by="qty/day")
grouped_day

# %%
fig, ax = plt.subplots()
fig.set_size_inches((4.5, 9.5))
xc, yc = utils.get_center_points(antof_grid)
i, j = 0, 0
between_x = xc[0][1] - xc[0][0]
between_y = yc[1][0] - yc[0][0]
labels = [False, False, False]
for xp in xc[0]:
    for yp in yc.T[0]:
        quad = utils.calc_quadrant(i, j, antof_grid[0].shape[1] - 1)
        xf = xp - between_x / 2
        yf = yp - between_y / 2
        group_freq = np.float16(grouped_day[grouped_day["group"] == quad]["qty/day"])[0] if quad in grouped_day["group"].values else 0
        color = "r" if group_freq > 0 else "g"
        color = "b" if group_freq > 0.5 else color
        label = ""
        if group_freq == 0 and not labels[0]:
            label = "Sin accidentes"
            labels[0] = True
        
        if group_freq > 0 and not labels[1] and label == "":
            label = "Accidentes por día"
            labels[1] = True

        if group_freq > 0.5 and not labels[2] and label == "":
            label = "Zona crítica"
            labels[2] = True

        ax.text(xp - 150, yp - 150, round(group_freq, 1), fontsize=7, alpha=0.8)
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
    ax, crs=alerts_grouped.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
)

plt.title("Accidentes por día por cuadrante")
plt.ylabel("Latitud")
plt.xlabel("Longitud")
plt.legend(fontsize=8, loc="upper left")
plt.show()

# %%
fig, ax = plt.subplots()
fig.set_size_inches((4.5, 9.5))

xc, yc = utils.get_center_points(antof_grid)
i, j = 0, 0
between_x = xc[0][1] - xc[0][0]
between_y = yc[1][0] - yc[0][0]
for xp in xc[0]:
    for yp in yc.T[0]:
        quad = utils.calc_quadrant(i, j, antof_grid[0].shape[1] - 1)
        ax.text(xp - 150, yp - 150, quad, fontsize=6, alpha=0.5)
        xf = xp - between_x / 2
        yf = yp - between_y / 2

        ax.fill_between(
            [xf, xf + between_x],
            yf,
            yf + between_y,
            alpha=(
                (
                    alerts_grouped["group"].value_counts()[
                        quad
                        if quad in alerts_grouped["group"].value_counts().index
                        else alerts_grouped["group"].value_counts().idxmin()
                    ]
                    - alerts_grouped["group"].value_counts().min()
                )
                / (
                    alerts_grouped["group"].value_counts().max()
                    - alerts_grouped["group"].value_counts().min()
                )
            ) ,
            color="r"
        )
        j += 1
    i += 1
    j = 0

cx.add_basemap(ax, crs=alerts_grouped.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

plt.title("Accidentes por cuadrante")
plt.ylabel("Latitud")
plt.xlabel("Longitud")
plt.show()

# %%
int(alerts_grouped.value_counts("group").reset_index()[alerts_grouped.value_counts("group").reset_index()["group"] == 132]["count"])

# %%
fig, ax = plt.subplots()
fig.set_size_inches((4.5, 9.5))

xc, yc = utils.get_center_points(antof_grid)
i, j = 0, 0
between_x = xc[0][1] - xc[0][0]
between_y = yc[1][0] - yc[0][0]
labels = [False, False, False]
for xp in xc[0]:
    for yp in yc.T[0]:
        quad = utils.calc_quadrant(i, j, antof_grid[0].shape[1] - 1)

        qty = (
            np.int16(
                alerts_grouped.value_counts("group").reset_index()[
                    alerts_grouped.value_counts("group").reset_index()["group"] == quad
                ]["count"]
            )[0]
            if quad
            in alerts_grouped.value_counts("group").reset_index()["group"].values
            else 0
        )

        color = "r" if qty > 0 else "g"
        color = "b" if qty > 60 else color
        label = ""
        if qty == 0 and not labels[0]:
            label = "Sin accidentes"
            labels[0] = True
        
        if qty > 0 and not labels[1] and label == "":
            label = "Accidentes totales en el periodo"
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
    ax, crs=alerts_grouped.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik
)

plt.title("Accidentes por cuadrante")
plt.ylabel("Latitud")
plt.xlabel("Longitud")
plt.legend(fontsize=8, loc="upper left")
plt.show()


# %%
fig, ax = plt.subplots()
fig.set_size_inches((4.5, 9.5))

xc, yc = utils.get_center_points(antof_grid)
i, j = 0, 0
between_x = xc[0][1] - xc[0][0]
between_y = yc[1][0] - yc[0][0]
for xp in xc[0]:
    for yp in yc.T[0]:
        quad = utils.calc_quadrant(i, j, antof_grid[0].shape[1] - 1)
        # ax.text(xp - 150, yp - 150, quad, fontsize=6, alpha=0.5)
        xf = xp - between_x / 2
        yf = yp - between_y / 2

        ax.fill_between(
            [xf, xf + between_x],
            yf,
            yf + between_y,
            alpha=(
                (
                    alerts_grouped["group"].value_counts()[
                        quad
                        if quad in alerts_grouped["group"].value_counts().index
                        else alerts_grouped["group"].value_counts().idxmin()
                    ]
                    - alerts_grouped["group"].value_counts().min()
                )
                / (
                    alerts_grouped["group"].value_counts().max()
                    - alerts_grouped["group"].value_counts().min()
                )
            ) ,
            color="r"
        )
        j += 1
    i += 1
    j = 0

cx.add_basemap(ax, crs=alerts_grouped.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

plt.title("Accidentes por cuadrante")
plt.ylabel("Latitud")
plt.xlabel("Longitud")
plt.show()

# %%
alerts_grouped

# %%
alerts_cleaned = alerts_grouped.drop(["street", "inicio", "fin", "x", "y", "geometry", "day"], axis=1)
alerts_cleaned

# %%
results = utils.xgb_classifier(alerts_cleaned, "happen", ohe=False)


# %%
xgb = results["model"]

# %%
xgb.predict(results["X_test"])

# %%
results["X_train"]

# %%
day_type = 1
hour = 7
minute = 30
week_day = 3
group = 76

# %%
# Preparing a sample with the same structure as X_train_happen
obj = pd.DataFrame(columns=results["X_train"].columns)
obj.loc[0] = 0  # Initialize all values to 0

probs = []

# Set desired features
obj["day_type"] = day_type
obj["hour"] = hour
obj["minute"] = minute
obj["week_day"] = week_day
obj["group"] = group

prob_happen = xgb.predict_proba(obj)
prob_happen


# %%
fig, ax = plt.subplots()
fig.set_size_inches((4.5, 9.5))
xc, yc = utils.get_center_points(antof_grid)
i, j = 0, 0
between_x = xc[0][1] - xc[0][0]
between_y = yc[1][0] - yc[0][0]
for xp in xc[0]:
    for yp in yc.T[0]:
        quad = utils.calc_quadrant(i, j, antof_grid[0].shape[1] - 1)
        xf = xp - between_x / 2
        yf = yp - between_y / 2
        obj["group"] = quad
        pred = xgb.predict_proba(obj)[0][1]
        ax.text(xp - 150, yp - 150, round(pred, 1), fontsize=7, alpha=0.8)
        ax.fill_between(
            [xf, xf + between_x],
            yf,
            yf + between_y,
            alpha=pred*0.7,
            color="r",
        )
        j += 1
    i += 1
    j = 0

cx.add_basemap(ax, crs=alerts_grouped.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)

plt.title("Accidentes por cuadrante")
plt.ylabel("Latitud")
plt.xlabel("Longitud")
plt.show()
    
