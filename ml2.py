from utils import utils
from analytics.grouper import Grouper
import pandas as pd
from analytics.ml import ML
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

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
    learning_rate=0.1,
    random_state=42,
    n_estimators=80,
    max_depth=20,
    gamma=0.8,
    colsample_bytree=0.7,
)

ml = ML(g.data, model, "happen", ["type"], True)
ml.generate_neg_simulated_data(extra_cols=x_vars)
ml.clean(x_vars, "happen")
ml.prepare_train()


params = {
    "learning_rate": [0.1, 0.2, 0.4, 0.8],
    "n_estimators": [80, 200],
    "max_depth": [20, 30, 40],
    "gamma": [0.8, 1, 1.2],
    "colsample_bytree": [0.7],
}

# Curva de Aprendizaje
train_sizes, train_scores, valid_scores = learning_curve(
    ml.model, ml.x_train, ml.y_train, cv=5
)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, valid_scores.mean(axis=1), label="Validation score")
plt.xticks(train_sizes, fontsize=10)
plt.yticks(np.arange(0, 1.0001, 0.1), fontsize=10)
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend(fontsize=11)
plt.title("Learning Curve")
plt.tight_layout()
plt.savefig("graph/learning_curve.png")
plt.show()
