from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from analytics.ml import ML
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

CONCEPTS = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]
x_vars = ["group", "hour", "day_type", "type", "week_day"]

alerts = pd.DataFrame(utils.load_data("alerts").data)
alerts = utils.update_timezone(alerts, "America/Santiago")

extra_cols = [
    "type",
    "geometry",
    "hour",
    "day_type",
    "week_day",
]

alerts = utils.extract_event(
    alerts,
    CONCEPTS,
    extra_col=extra_cols,
)

g = Grouper(alerts)
g.group((10, 20))

model = XGBClassifier(
    learning_rate=0.03,
    random_state=42,
    n_estimators=50,
    max_depth=5,
    gamma=0.2,
    colsample_bytree=0.7,
)

ml = ML(g.data, model, "happen", ["type"], True)
ml.generate_neg_simulated_data(extra_cols=x_vars)
ml.clean(x_vars, "happen")
ml.prepare_train()


params = {
    "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "n_estimators": [10, 30, 50, 80, 100, 150],
    "max_depth": [2, 5, 8, 12, 15, 20],
    "gamma": [0.2, 0.4, 0.6, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
}


gs = GridSearchCV(model, param_grid=params, cv=5)
gs.fit(ml.x_train, ml.y_train)

print("\nPARAMS: ")
print(params)
print("\nBEST_PARAMS: ")
print(gs.best_params_)
print("\nBEST_METRICS: ")
print(gs.best_score_)

# # Curva de Aprendizaje
# train_sizes, train_scores, valid_scores = learning_curve(
#     ml.model, ml.x_train, ml.y_train, cv=5
# )
# plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
# plt.plot(train_sizes, valid_scores.mean(axis=1), label="Validation score")
# plt.xlabel("Training Size")
# plt.ylabel("Score")
# plt.legend()
# plt.show()
