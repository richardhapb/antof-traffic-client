from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
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
import numpy as np

CONCEPTS = ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"]


def init_mlflow():
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
        self.oe = None
        self.no_events = None
        self.x = None
        self.y = None

        if self.data is not None:
            self.data[column_y] = 1

    @staticmethod
    def convert_dataset_to_ohe(data: pd.DataFrame, categories: list):
        labeled = []
        onehot = {}
        for c in categories:
            onehot[c] = OneHotEncoder(handle_unknown="ignore")
            oht = onehot[c].fit_transform(data[[c]])

            labeled.append(
                pd.DataFrame(
                    oht.toarray(), columns=onehot[c].get_feature_names_out()
                ).reset_index(drop=True)
            )

        return pd.concat([data.drop(categories, axis=1), *labeled], axis=1)

    def generate_neg_simulated_data(self, extra_cols: list, geodata: str = "group"):
        events2 = self.data.copy()
        events2["happen"] = 1

        if not isinstance(events2["inicio"].iloc[0], np.int64):
            events2["inicio"] = np.int64(pd.to_numeric(events2["inicio"]) / 1_000_000)

        step = np.int64(60_000 * 5)

        min_tms = events2["inicio"].min()
        max_tms = events2["inicio"].max()

        intervals = np.arange(min_tms, max_tms + step, step)

        events2["interval"] = ((events2["inicio"]) // step) * step

        allgroups = events2[geodata].unique()
        alltypes = events2["type"].unique()

        combinations = pd.MultiIndex.from_product(
            [intervals, allgroups, alltypes], names=["interval", geodata, "type"]
        ).to_frame(index=False)

        event_combinations = events2[["interval", geodata, "type"]]
        event_combinations["happen"] = 1

        merged = pd.merge(
            combinations,
            event_combinations,
            on=["interval", geodata, "type"],
            how="left",
        )

        merged = merged.sample(events2.shape[0], replace=False, random_state=42)
        merged["happen"] = merged["happen"].fillna(0).astype(int)

        merged["inicio"] = merged["interval"]

        result = merged[["inicio", geodata, "type", "happen"]]

        if not isinstance(self.data["inicio"].iloc[0], np.int64):
            result["inicio"] = pd.to_datetime((result["inicio"] * 1_000_000), unit="ns")
        else:
            result["inicio"] = result["inicio"]

        self.no_events = pd.DataFrame(result)
        self.no_events = utils.extract_event(
            self.no_events, CONCEPTS, extra_cols + ["happen"]
        )

    def balance_day_type(self):
        # Balance no events and events, for weekend and workdays
        if (
            "day_type" not in self.data.columns
            or "day_type" not in self.no_events.columns
        ):
            raise ValueError("There are not day type data in dataset")

        print("BEFORE")

        print(self.data.day_type.value_counts())
        print(self.no_events.day_type.value_counts())

        days0 = self.data[(self.data.day_type == "f") | (self.data.day_type == 0)]
        days1 = self.data[(self.data.day_type == "s") | (self.data.day_type == 1)]
        q_days0 = days0.shape[0]
        q_days1 = days1.shape[0]

        diff = q_days1 - q_days0

        self.data = pd.concat(
            [days1, days0.sample(q_days0 + diff, replace=True, random_state=42)]
        )

        days0 = self.no_events[
            (self.no_events.day_type == "f") | (self.no_events.day_type == 0)
        ]
        days1 = self.no_events[
            (self.no_events.day_type == "s") | (self.no_events.day_type == 1)
        ]

        self.no_events = pd.concat(
            [
                days1.sample(q_days1, replace=True, random_state=42),
                days0.sample(q_days0 + diff, replace=True, random_state=42),
            ]
        )

        print("AFTER")
        print(self.data.day_type.value_counts())
        print(self.no_events.day_type.value_counts())

    def clean(self, columns_x, column_y):
        self.columns_x = columns_x
        self.column_y = column_y

        # Clean the data
        self.data = self.data[columns_x + [column_y]]
        if self.no_events is not None:
            self.no_events = self.no_events[columns_x + [column_y]]

    def prepare(self, no_features=None):
        if self.data is None:
            raise ValueError("Not have data for model generation")

        total_events = (
            pd.concat([self.data, self.no_events])
            if self.no_events is not None
            else self.data.copy()
        )

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

    def train(self):
        self.x = self.data_labeled.drop(self.column_y, axis=1)
        self.y = self.data_labeled[self.column_y]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
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

        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict(data)

    def predict_proba(self, data):
        if self.data_labeled is None:
            self.train()

        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

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

    def confusion_matrix(self):
        return confusion_matrix(
            self.y_test,
            self.model.predict(self.x_test),
            labels=self.y_train.unique(),
        )

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
        obj = obj.copy()
        if self.data_labeled is None:
            self.train()
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))
        xc, yc = grouper.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]

        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = grouper.calc_quadrant(i, j)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                if self.ohe and "group" in self.categories:
                    obj["group_" + str(quad)] = 1
                else:
                    obj["group"] = quad
                pred = self.predict_proba(obj)[0][1]
                ax.text(xp - 250, yp - 150, round(pred, 1), fontsize=6, alpha=0.8)
                ax.fill_between(
                    [xf, xf + between_x],
                    yf,
                    yf + between_y,
                    alpha=pred * 0.7,
                    color="r",
                )
                if self.ohe and "group" in self.categories:
                    obj["group_" + str(quad)] = 0
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
                input_example=self.x_train.head(),
                registered_model_name=type(self.model).__name__,
            )

        model_params = self.model.get_params()
        mlflow.log_params({k: v for k, v in model_params.items() if v is not None})

        scores = self.cross_validation()
        cm = self.confusion_matrix()

        for idx, score in enumerate(scores):
            mlflow.log_metric(f"cv_score_fold_{idx}", score)
        metrics = self.metrics()

        cats = ["positive", "negative"]
        for c in range(len(cm)):
            metrics.update({"cm_true_" + cats[c]: cm[c][0 + c]})
            metrics.update({"cm_false_" + cats[c]: cm[c][1 - c]})

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)

    def prepare_train(self):
        self.prepare()
        self.train()

    def ordinal_encoder(self, categories: list):
        if self.data is None:
            raise ValueError("Data is not loaded")
        self.oe = {}
        for cat in categories:
            if cat not in self.data.columns:
                raise ValueError(cat + " is not in data")
            self.oe[cat] = OrdinalEncoder()
            self.data[cat] = self.oe[cat].fit_transform(self.data[[cat]])
