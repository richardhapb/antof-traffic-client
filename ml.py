from analytics.ml import ML, init_mlflow
from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

init_mlflow()

print("\n\n\n\n\n\n")

ROUTES_NAMES = [
    "Costanera",
    "Av Argentina - Iquique - Pedro Aguirre Cerda",
    "Av. Argentina - Andrés Sabella - Antonio Rendic - Huamachuco - Pedro Aguirre Cerda",
    "Av. Argentina - Circunvalación - Bonilla - Pedro Aguirre Cerda",
]

GEODATA = "group"

CONCEPTS = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]

alerts = pd.DataFrame(utils.load_data("alerts").data)
alerts = utils.update_timezone(alerts, "America/Santiago")

alerts = utils.extract_event(
    alerts,
    CONCEPTS,
    ["type", "geometry", "hour", "day_type", "week_day", "day"]
    + (["street"] if GEODATA == "street" else []),
)


print(alerts[alerts["type"] == "ACCIDENT"].shape[0])

g = Grouper(alerts)


def predict_route(ml, initial_params, routes, **kwargs):
    print(
        ("\nHASHED DATA\n" if ml.hash else "\nONE HOT ENCODED DATA\n") + f"for {kwargs}"
    )

    obj = pd.DataFrame(columns=ml.x_train.columns)
    obj.loc[0] = 0

    for k, v in initial_params.items():
        obj[k] = v

    if ml.hash:
        for k, v in kwargs.items():
            result = ml.encode([str(v)], k)
            for r in range(len(result[0])):
                obj[k + "_" + str(r)] = 0
            if k in obj:
                del obj[k]

            obj[k + "_" + str(int(result[0][result[0].argmax()]))] = 1
    elif ml.ohe:
        for k, v in kwargs.items():
            for c in [e for e in ml.x_train.columns if k + "_" in e]:
                obj[c] = 0
            if k in obj:
                del obj[k]

            result = ml.encode([str(v)], k)
            obj[k + "_" + str(v)] = 1

    response = []
    if GEODATA == "group":
        i = 0
        for route in routes:
            print(
                "\n—————————————————————————————————————————————————————————————————\n"
            )
            print(f"Ruta: {ROUTES_NAMES[i]}")
            i += 1
            probs = []
            obj2 = pd.concat([obj] * len(route), ignore_index=True)
            obj2["group"] = route
            probs.append(ml.predict_proba(obj2))
            response.append(probs)

            print(f"Ruta: {route}")
            print("\nProbabilities:\n")
            print(probs)
            print(f"\nAverage probability: {np.average(np.array(probs)[0][:, 1])}\n")

    for route in routes if len(routes) > 0 and isinstance(routes[0][0], str) else []:
        probs = []
        print(" -> ".join(route) + "\n")

        for column in ml.x_train.columns:
            if column.split("_")[0] not in " ".join(ml.categories):
                obj[column] = initial_params[column]

        if ml.hash:
            for street in route:
                result = ml.encode([street], "street")
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 1
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 0
        elif ml.ohe:
            streets = ml.onehot["street"].get_feature_names_out()

            for street in route:
                result = ml.encode([street], "street")
                obj[streets[int(result[0][result[0].argmax()])]] = 1
                obj[streets[int(result[0][result[0].argmax()])]] = 0

        probs.append(ml.predict_proba(obj)[0])
        response.append(probs)

        print("\nProbabilities\n")
        print(" -> ".join([str(round(p[1], 3)) for p in probs]))
        print("\nAverage probability: ", round((sum(probs) / len(probs))[1], 3))
        print("\n—————————————————————————————————————————————————————————————————\n")

    if not routes:
        print("\nProbabilities\n")
        print(kwargs)
        response = ml.predict_proba(obj)
        print(response[0])

    return response


if GEODATA == "group":
    g.group((10, 20), CONCEPTS)
    g.filter_by_group_time(60, True)
    fig = g.plot_with_numbers()
    fig.savefig("graph/groups_with_numbers.png")

model = XGBClassifier(
    learning_rate=0.1,
    random_state=42,
    n_estimators=80,
    max_depth=20,
    gamma=0.8,
    colsample_bytree=0.7,
)

# model = RandomForestClassifier(
#     random_state=42,
#     n_estimators=100,
#     max_depth=10,
#     max_features=2,
#     max_leaf_nodes=8,
#     min_samples_leaf=2,
# )

route1 = [
    "Av. República de Croacia",  # No lo encuentra
    "Av. Balmaceda",
    "Av. Séptimo de Línea",
    "Av. Edmundo Pérez Zujovic",
]
route2 = [
    "Av. Argentina",
    "Av. Iquique",
    "El Yodo",
    "Nicolás Tirado",
    "Av. Pedro Aguirre Cerda",
]

# Costanera
route1_group = [22, 42, 62, 81, 82, 83, 84, 103, 104, 105, 87, 107, 89, 90, 110]

# Av Argentina - Iquique - Pedro Aguirre Cerda
route2_group = [60, 61, 81, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# Av. Argentina - Andrés Sabella - Antonio Rendic - Huamachuco - Pedro Aguirre Cerda
route3_group = [60, 61, 81, 101, 102, 103, 123, 124, 125, 144, 145, 127, 108, 109, 110]

# Av. Argentina - Circunvalación - Bonilla - Pedro Aguirre Cerda
route4_group = [60, 61, 81, 101, 121, 122, 142, 143, 144, 145, 127, 108, 109, 110]

if GEODATA == "street":
    routes = [route1, route2]
elif GEODATA == "group":
    routes = [route1_group, route2_group, route3_group, route4_group]


x_vars = [GEODATA, "hour", "day_type", "type", "week_day", "day"]
categories = ["type"]

ORDINAL_ENCODER = False

ml = ML(
    g.data if GEODATA == "group" else alerts,
    model,
    column_y="happen",
    ohe=True,
    categories=categories,
)

if ORDINAL_ENCODER:
    ml.ordinal_encoder(categories=[GEODATA])
ml.generate_neg_simulated_data(
    extra_cols=x_vars,
    geodata=GEODATA,
)
ml.clean(x_vars, "happen")
ml.prepare_train()

geodata_element = 72

initial_params = {
    "day_type": 1,
    "hour": 7,
    "week_day": 1,
    "day": 20,
    # GEODATA: geodata_element,
}

print("\nINITIAL PARAMS:\n")
for i in initial_params.items():
    print(i)


obj = pd.DataFrame(columns=ml.x_train.columns)
obj.loc[0] = 0

for k, v in initial_params.items():
    obj[k] = v

obj["type_ACCIDENT"] = 1

if "group" in obj:
    del obj["group"]

fig = ml.plot_by_quad(g, obj)
if "group" in categories:
    fig.savefig("graph/quad_with_probs_ohe.png")
else:
    fig.savefig("graph/quad_with_probs_linear.png")

print("\n—————————————————————————————————————————————————————————————————\n")

type_event = "ACCIDENT"

probs = []
if "group" in categories:
    for route in routes:
        pr = []
        for r in route:
            p = predict_route(ml, initial_params, [], type=type_event, group=r)
            pr.append(p)
        probs.append(pr)
        print("\nAverage: \n")
        print(np.average(np.array(pr[0]).ravel().reshape(-1, 2)[:, 1]))
        print("\n—————————————————————————————————————————————————————————————————\n")
else:
    probs = predict_route(ml, initial_params, routes, type=type_event)

print("\n—————————————————————————————————————————————————————————————————\n")

cm = ml.confusion_matrix()

print("Confusion matrix:\n")
print(cm)

ml.log_model_params(
    **initial_params,
    avg_pos_probs=np.average(np.array(probs[0]).ravel().reshape(-1, 2)[:, 1]),
    type_event=type_event,
    hash_encode=ml.hash,
    ohe=ml.ohe,
    sample=ml.data.shape,
    ordinal_encoder=ORDINAL_ENCODER,
    sample_no_events=ml.no_events.shape,
    geodata=GEODATA,
    geodata_element=geodata_element,
    categories=categories,
)

# ml.ohe = False
# ml.hash = True
# ml.prepare_train()

# cm = ml.confusion_matrix()
# probs = predict_route(ml, initial_params, routes, type=type_event)

# print("Confusion matrix:\n")
# print(cm)

# ml.log_model_params(
#     **initial_params,
#     avg_pos_probs=np.average(np.array(probs[0]).ravel().reshape(-1, 2)[:, 1]),
#     type_event=type_event,
#     hash_encode=ml.hash,
#     ohe=ml.ohe,
#     sample=ml.data.shape,
#     ordinal_encoder=ORDINAL_ENCODER,
#     sample_no_events=ml.no_events.shape,
#     geodata=GEODATA,
#     geodata_element=geodata_element,
#     categories=categories,
# )
