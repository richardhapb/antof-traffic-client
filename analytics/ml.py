import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import clone
import pandas as pd
from analytics.grouper import Grouper
from utils import utils
import matplotlib.pyplot as plt
import contextily as cx
import mlflow
from mlflow.models import infer_signature
from functools import wraps
from sklearn import base

mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("AntofTraffic")


def mlflow_logger(func):
    @wraps(func)
    def mlflow_wrapper(self, *args, **kwargs):
        with mlflow.start_run():
            result = func(self, *args, **kwargs)
            return result

    return mlflow_wrapper


class ML:
    def __init__(
        self, data, model, column_y=None, categories=None, ohe=False, hash=False
    ):
        self.data = data
        self.model = clone(model)
        self.columns_x = [column for column in self.data.columns if column != column_y]
        self.column_y = column_y
        self.categories = categories
        self.ohe = ohe
        self.hash = hash
        self.hasher = None
        self.data_labeled = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.onehot = None

        probs_type = {
            "ACCIDENT": 1 - data["type"].value_counts(normalize=True)["ACCIDENT"],
            "JAM": 1 - data["type"].value_counts(normalize=True)["JAM"],
            "HAZARD": 1 - data["type"].value_counts(normalize=True)["HAZARD"],
            "ROAD_CLOSED": 1 - data["type"].value_counts(normalize=True)["ROAD_CLOSED"],
        }

        t = sum(probs_type.values())
        for k, v in probs_type.items():
            probs_type[k] = v / t

        probs_day_type = {"s": 2 / 7, "f": 5 / 7}

        self.simulated_data = {
            "street": self.data["street"] if "street" in self.data.columns else None,
            "day_type": np.random.choice(
                list(probs_day_type.keys()),
                len(self.data),
                p=list(probs_day_type.values()),
            )
            if "day_type" in self.data.columns
            else None,
            "type": np.random.choice(
                list(probs_type.keys()), len(self.data), p=list(probs_type.values())
            )
            if "type" in self.data.columns
            else None,
            "group": np.random.randint(
                self.data["group"].min(), self.data["group"].max(), len(self.data)
            )
            if "group" in data.columns
            else None,
            "day": np.random.randint(
                self.data["day"].min(), self.data["day"].max() + 1, len(self.data)
            )
            if "day" in self.data.columns
            else None,
            "week_day": np.random.randint(
                self.data["week_day"].min(), self.data["week_day"].max(), len(self.data)
            )
            if "week_day" in self.data.columns
            else None,
            "hour": np.random.randint(
                self.data["hour"].min(), self.data["hour"].max(), len(self.data)
            )
            if "hour" in self.data.columns
            else None,
            "minute": np.random.randint(
                self.data["minute"].min(), self.data["minute"].max(), len(self.data)
            )
            if "minute" in self.data.columns
            else None,
            "happen": [0] * len(self.data),
        }

    def clean(self, columns_x, column_y):
        self.columns_x = columns_x
        self.column_y = column_y

        # Clean the data
        for column in self.data.columns:
            if column not in self.columns_x and column != self.column_y:
                self.data = self.data.drop(column, axis=1)

    def train(self, no_features=None):
        no_events = pd.DataFrame()
        for column in self.columns_x:
            no_events[column] = self.simulated_data[column]

        if self.column_y == "happen":
            self.data["happen"] = 1
            no_events["happen"] = 0

        total_events = pd.concat([self.data, no_events], axis=0)

        if "day_type" in total_events.columns:
            total_events["day_type"] = total_events["day_type"].map({"f": 0, "s": 1})

        if self.ohe:
            labeled = []
            self.onehot = {}
            for c in self.categories:
                self.onehot[c] = OneHotEncoder(handle_unknown="ignore")
                oht = self.onehot[c].fit_transform(total_events[[c]])

                labeled.append(
                    pd.DataFrame(
                        oht.toarray(), columns=self.onehot[c].get_feature_names_out()
                    ).reset_index(drop=True)
                )

            self.data_labeled = pd.concat(
                [
                    total_events.drop(self.categories, axis=1).reset_index(drop=True),
                    *labeled,
                ],
                axis=1,
            )
        elif self.hash:
            labeled = []
            self.hasher = {}
            no_features = no_features if no_features else 5
            for c in self.categories:
                self.hasher[c] = FeatureHasher(
                    n_features=no_features, input_type="string"
                )
                # Eliminar valores nulos y convertir todos los valores a strings
                filtered_events = [str(s) for s in total_events[c] if pd.notnull(s)]
                hashed = self.hasher[c].fit_transform([[s] for s in filtered_events])

                labeled.append(
                    pd.DataFrame(
                        hashed.toarray(),
                        columns=[f"{c}_{i}" for i in range(no_features)],
                    ).reset_index(drop=True)
                )

            self.data_labeled = pd.concat(
                [
                    total_events.drop(self.categories, axis=1).reset_index(drop=True),
                    *labeled,
                ],
                axis=1,
            )
        else:
            self.data_labeled = total_events.copy()

        x = self.data_labeled.drop(self.column_y, axis=1)
        y = self.data_labeled[self.column_y]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        self.model.fit(self.x_train, self.y_train)

    def cross_validation(self):
        if self.data_labeled is None:
            self.train()

        x = self.data_labeled.drop(self.column_y, axis=1)
        y = self.data_labeled[self.column_y]

        scores = cross_val_score(self.model, x, y, cv=10)

        return scores

    def predict(self, data):
        if self.data_labeled is None:
            self.train()

        return self.model.predict(data)

    def predict_proba(self, data):
        if self.data_labeled is None:
            self.train()

        return self.model.predict_proba(data)

    def select_best_params(self, param_grid):
        if self.data_labeled is None:
            self.train()

        return self.model.best_params_

    def metrics(self):
        y_pred = self.model.predict(self.x_test)

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
        }

        return metrics

    def encode(self, data, category):
        if self.data_labeled is None:
            self.train()

        if self.ohe:
            return (
                self.onehot[category]
                .transform(pd.DataFrame([data], columns=[category]))
                .toarray()
            )
        elif self.hash:
            return self.hasher[category].transform([[s] for s in data]).toarray()

    def plot_by_quad(self, grouper: Grouper, obj: pd.DataFrame):
        if "group" not in self.data.columns:
            raise ValueError("The data must have a 'group' column")
        if self.data_labeled is None:
            self.train()
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))
        xc, yc = utils.get_center_points(grouper.grid)
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]
        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = utils.calc_quadrant(i, j, grouper.grid[0].shape[1] - 1)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                obj["group"] = quad
                pred = self.predict_proba(obj)[0][1]
                ax.text(xp - 150, yp - 150, round(pred, 1), fontsize=7, alpha=0.8)
                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=pred * 0.7,
                    color="r",
                )
                j += 1
            i += 1
            j = 0

        cx.add_basemap(
            ax,
            crs=grouper.data.crs.to_string(),
            source=cx.providers.OpenStreetMap.Mapnik,
        )

        ax.set_title("Accidentes por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    @mlflow_logger
    def log_model_params(self, **params):
        signature = infer_signature(self.x_train, self.model.predict(self.x_train))

        if isinstance(self.model, base.BaseEstimator):
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="waze_data",
                signature=signature,
                input_example=self.x_train,
                registered_model_name=type(self.model).__name__,
            )

        model_params = self.model.get_params()
        mlflow.log_params({k: v for k, v in model_params.items() if v is not None})

        scores = self.cross_validation()

        for idx, score in enumerate(scores):
            mlflow.log_metric(f"cv_score_fold_{idx}", score)

        metrics = self.metrics()

        mlflow.log_metrics(metrics)

        mlflow.log_params(params)
