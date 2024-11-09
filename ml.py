from analytics.ml import ML
from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

g = Grouper()


def predict_route(ml, initial_params, routes, **kwargs):
    print("\nHASHED DATA\n" if ml.hash else "\nONE HOT ENCODED DATA\n")

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
        for route in routes:
            probs = []
            obj2 = pd.concat([obj] * len(route), ignore_index=True)
            obj2["group"] = route
            probs.append(ml.predict_proba(obj2))
            response.append(probs)
            print(
                "\n—————————————————————————————————————————————————————————————————\n"
            )

            print(f"Route: {route}")
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


CONCEPTS = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]

alerts = pd.DataFrame(utils.load_data("alerts").data)
alerts = utils.update_timezone(alerts, "America/Santiago")

GEODATA = "group"

alerts = utils.extract_event(
    alerts,
    CONCEPTS,
    ["type", "geometry", "hour", "day_type", "week_day"]
    + (["street"] if GEODATA == "street" else []),
)

if GEODATA == "group":
    g.group(alerts, (15, 30), CONCEPTS)
    fig = g.plot_with_numbers()
    fig.savefig("graph/groups_with_numbers.png")

model = XGBClassifier(
    learning_rate=0.03,
    random_state=42,
    n_estimators=50,
    max_depth=5,
    gamma=0.2,
    colsample_bytree=0.7,
)

# model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)

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

route1_group = [
    45,
    60,
    75,
    89,
    90,
    105,
    119,
    133,
    134,
    148,
    162,
    177,
    191,
    205,
    218,
    219,
    232,
    247,
    260,
    274,
    287,
    302,
]

route2_group = [
    46,
    61,
    76,
    91,
    106,
    120,
    121,
    135,
    150,
    163,
    178,
    193,
    206,
    220,
    233,
    234,
    247,
    261,
    275,
    289,
    303,
]

if GEODATA == "street":
    routes = [route1, route2]
elif GEODATA == "group":
    routes = [route1_group, route2_group]


x_vars = [GEODATA, "hour", "day_type", "type", "week_day"]
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
    "hour": 10,
    "week_day": 2,
    GEODATA: geodata_element,
}

print("\nINITIAL PARAMS:\n")
for i in initial_params.items():
    print(i)


obj = pd.DataFrame(columns=ml.x_train.columns)
obj.loc[0] = 0

for k, v in initial_params.items():
    obj[k] = v

if "group" in obj:
    del obj["group"]

fig = ml.plot_by_quad(g, obj)
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

# probs = predict_route(ml, initial_params, routes, type=type_event)

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
