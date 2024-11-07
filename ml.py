from analytics.ml import ML
from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from xgboost import XGBClassifier


def predict_route(ml, initial_params, routes):
    print("\nHASHED DATA\n" if ml.hash else "\nONE HOT ENCODED DATA\n")

    print("Cross validation\n")
    print(ml.cross_validation())
    print("\n—————————————————————————————————————————————————————————————————\n")
    print("Metrics\n")
    print(ml.metrics())
    print("\n—————————————————————————————————————————————————————————————————\n")

    for route in routes:
        probs = []
        print(" -> ".join(route) + "\n")
        obj = pd.DataFrame(columns=ml.x_train.columns)
        obj.loc[0] = 0

        for column in ml.x_train.columns:
            if "street" not in column:
                obj[column] = initial_params[column]

        if ml.hash:
            for street in route:
                result = ml.encode([street], "street")
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 1
                probs.append(ml.predict_proba(obj)[0])
                obj["street_" + str(int(result[0][result[0].argmax()]))] = 0
        else:
            streets = ml.onehot["street"].get_feature_names_out()

            for street in route:
                result = ml.encode([street], "street")
                obj[streets[int(result[0][result[0].argmax()])]] = 1
                probs.append(ml.predict_proba(obj)[0])
                obj[streets[int(result[0][result[0].argmax()])]] = 0

        print("\nProbabilities\n")
        print(" -> ".join([str(round(p[1], 3)) for p in probs]))
        print("\nAverage probability: ", round((sum(probs) / len(probs))[1], 3))
        print("\n—————————————————————————————————————————————————————————————————\n")


alerts = pd.DataFrame(utils.load_data("alerts").data)
alerts = utils.extract_event(alerts, ["ACCIDENT"]).drop("uuid", axis=1)

alerts = alerts.dropna(subset=["street"])

g = Grouper()
g.group(alerts, (10, 20), "ACCIDENT")

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
    "week_day": 3,
    "group": 76,
}

x_vars = ["group", "week_day", "hour", "day_type"]
categories = []

ml_ohe = ML(g.data, xgb, column_y="happen", ohe=True, categories=categories)
ml_ohe.clean(x_vars, "happen")
ml_ohe.train()

ml_hash = ML(g.data, xgb, column_y="happen", hash=True, categories=categories)
ml_hash.clean(x_vars, "happen")
ml_hash.train()

# predict_route(ml_ohe, initial_params, routes)
# predict_route(ml_hash, initial_params, routes)


obj = pd.DataFrame(columns=ml_ohe.x_train.columns)
obj.loc[0] = 0

# for column in ml_ohe.x_train.columns:
#     if "street" not in column:
#         obj[column] = initial_params[column]

# streets = ml_ohe.onehot["street"].get_feature_names_out()
# for column in streets:
#     obj[column] = 0

# for street in ml_ohe.data["street"].unique():
#     result = ml_ohe.encode([street], "street")
#     if "street_" + street in streets:
#         obj[streets[int(result[0][result[0].argmax()])]] = 1

fig = ml_ohe.plot_by_quad(g, obj)
fig.savefig("graph/accidents_by_quad_ohe.png")


obj = pd.DataFrame(columns=ml_hash.x_train.columns)
obj.loc[0] = 0

# for column in ml_hash.x_train.columns:
#     if "street" not in column:
#         obj[column] = initial_params[column]

# result = ml_hash.encode([initial_params["street"]], "street")

# obj["street_" + str(int(result[0][result[0].argmax()]))] = 1

fig = ml_hash.plot_by_quad(g, obj)
fig.savefig("graph/accidents_by_quad_hash.png")
