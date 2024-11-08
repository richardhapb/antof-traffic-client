from analytics.ml import ML
from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from xgboost import XGBClassifier


def predict_route(ml, initial_params, routes, **kwargs):
    print("\nHASHED DATA\n" if ml.hash else "\nONE HOT ENCODED DATA\n")

    print("Cross validation\n")
    print(ml.cross_validation())
    print("\n—————————————————————————————————————————————————————————————————\n")
    print("Metrics\n")
    print(ml.metrics())
    print("\n—————————————————————————————————————————————————————————————————\n")

    obj = pd.DataFrame(columns=ml.x_train.columns)
    obj.loc[0] = 0

    if ml.hash:
        for k, v in kwargs.items():
            result = ml.encode([str(v)], k)
            for r in range(len(result[0])):
                obj[k + "_" + str(r)] = 0

            obj[k + "_" + str(int(result[0][result[0].argmax()]))] = 1
    elif ml.ohe:
        for k, v in kwargs.items():
            for c in [e for e in ml.x_train.columns if k in e]:
                obj[c] = 0

            result = ml.encode([str(v)], k)
            obj[k + "_" + str(v)] = 1

    probs = []
    for route in routes:
        print(" -> ".join(route) + "\n")

        for column in ml.x_train.columns:
            if column.split("_")[0] not in " ".join(ml.categories):
                obj[column] = initial_params[column]

        if ml.hash:
            for street in route:
                result = ml.encode([street], "street")
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 1
                probs.append(ml.predict_proba(obj)[0])
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 0
        elif ml.ohe:
            streets = ml.onehot["street"].get_feature_names_out()

            for street in route:
                result = ml.encode([street], "street")
                obj[streets[int(result[0][result[0].argmax()])]] = 1
                probs.append(ml.predict_proba(obj)[0])
                obj[streets[int(result[0][result[0].argmax()])]] = 0

        probs.append(ml.predict_proba(obj)[0])

        print("\nProbabilities\n")
        print(" -> ".join([str(round(p[1], 3)) for p in probs]))
        print("\nAverage probability: ", round((sum(probs) / len(probs))[1], 3))
        print("\n—————————————————————————————————————————————————————————————————\n")

    if not routes:
        print("\nProbabilities\n")
        probs = ml.predict_proba(obj)
        print(probs[0])

    return probs


CONCEPTS = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]

alerts = pd.DataFrame(utils.load_data("alerts").data)
alerts = utils.update_timezone(alerts, "America/Santiago")

GEODATA = "street"

alerts = utils.extract_event(
    alerts, CONCEPTS, ["type", "geometry"] + (["street"] if GEODATA == "street" else [])
)

g = Grouper()
if GEODATA == "group":
    g.group(alerts, (10, 20), CONCEPTS)

xgb = XGBClassifier(
    learning_rate=0.03,
    random_state=42,
    n_estimators=50,
    max_depth=5,
    gamma=0.2,
    colsample_bytree=0.7,
)

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
route1_return = [
    "Av. Edmundo Pérez Zujovic",
    "Av. Séptimo de Línea",
    "Av. Balmaceda",
    "Av. República de Croacia",  # No lo encuentra
]
route2_return = [
    "Av. Pedro Aguirre Cerda",
    "Nicolás Tirado",
    "El Yodo",
    "Av. Iquique",
    "Av. Argentina",
]

routes = [route1, route2, route1_return, route2_return]


x_vars = [GEODATA, "hour", "day_type", "type"]
categories = ["type"]

ORDINAL_ENCODER = True

ml = ML(
    g.data if GEODATA == "group" else alerts,
    xgb,
    column_y="happen",
    ohe=True,
    categories=categories,
)
if ORDINAL_ENCODER:
    ml.ordinal_encoder(categories=[GEODATA])
ml.generate_neg_simulated_data(extra_cols=["type", GEODATA], geodata=GEODATA)
ml.clean(x_vars, "happen")
ml.prepare_train()

print(ml.oe["street"].get_feature_names_out())

geodata_element = 60

initial_params = {"day_type": 1, "hour": 7, GEODATA: geodata_element}

print("\nINITIAL PARAMS:\n")
for i in initial_params.items():
    print(i)

print("\n—————————————————————————————————————————————————————————————————\n")

type_event = "ACCIDENT"

probs = predict_route(ml, initial_params, [], type=type_event)

print("\n—————————————————————————————————————————————————————————————————\n")

cm = ml.confusion_matrix()

print(cm)

ml.log_model_params(
    **initial_params,
    probabilities=probs,
    type_event=type_event,
    hash_encode=ml.hash,
    ohe=ml.ohe,
    sample=ml.data.shape,
    ordinal_encoder=ORDINAL_ENCODER,
    sample_no_events=ml.no_events.shape,
    confusion_matrix=cm,
    each_x_min=60 * 12,
    geodata=GEODATA,
    geodata_element=geodata_element,
)

ml.ohe = False
ml.hash = True
ml.prepare_train()

probs = predict_route(ml, initial_params, [], type="JAM")

ml.log_model_params(
    **initial_params,
    probabilities=probs,
    type_event=type_event,
    hash_encode=ml.hash,
    ohe=ml.ohe,
    sample=ml.data.shape,
    ordinal_encoder=ORDINAL_ENCODER,
    sample_no_events=ml.no_events.shape,
    confusion_matrix=cm,
    each_x_min=60 * 12,
    geodata=GEODATA,
    geodata_element=geodata_element,
)
