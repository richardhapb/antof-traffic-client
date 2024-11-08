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
alerts = utils.extract_event(alerts, CONCEPTS, ["type"]).drop("uuid", axis=1)

alerts = alerts.dropna(subset=["street"])

g = Grouper()
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

initial_params = {
    "day_type": 1,
    "hour": 7,
}

x_vars = ["group", "hour", "day_type", "type"]
categories = ["type", "group"]

ml_ohe = ML(g.data, xgb, column_y="happen", ohe=True, categories=categories)
ml_ohe.clean(x_vars, "happen")
ml_ohe.train()

# ml_hash = ML(g.data, xgb, column_y="happen", hash=True, categories=categories)
# ml_hash.clean(x_vars, "happen")
# ml_hash.train()


print("\nINITIAL PARAMS:\n")
for i in initial_params.items():
    print(i)

print("\n—————————————————————————————————————————————————————————————————\n")

probs = predict_route(ml_ohe, initial_params, [], type="JAM", group=60)

g.data.isna()
# ml_ohe.log_model_params(**initial_params, probabilities=probs)
# predict_route(ml_hash, initial_params, [], type="JAM", group=60)
