import os
from functools import wraps
from typing import Dict, List, cast

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from sklearn.base import clone
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

import mlflow
from mlflow.models import infer_signature
from utils import utils
from utils.utils import logger


MODEL_NAME = "XGBClassifier"


def init_mlflow():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
    mlflow.set_tracking_uri(uri=mlflow_uri)
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
        self,
        data: GeoDataFrame | pd.DataFrame,
        model: XGBClassifier,
        column_y: str | None = None,
        categories: List | None = None,
        ohe: bool = False,
        hash: bool = False,
    ):
        self.data: GeoDataFrame | pd.DataFrame = data
        self.model: XGBClassifier = cast(XGBClassifier, clone(model))
        self.columns_x: List = [column for column in self.data.columns if column != column_y]
        self.column_y: str | None = column_y
        self.categories: List | None = categories
        self.ohe: bool = ohe
        self.hash: bool = hash
        self.hasher: Dict[str, FeatureHasher] | None = None
        self.data_labeled: GeoDataFrame | pd.DataFrame | None = None
        self.x_train: pd.DataFrame | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_train: pd.DataFrame | None = None
        self.y_test: pd.DataFrame | None = None
        self.onehot: Dict[str, OneHotEncoder] | None = None
        self.oe: Dict[str, OrdinalEncoder] | None = None  # Ordinal Encoder
        self.total_events: pd.DataFrame | GeoDataFrame | None = None
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

        if self.data is not None:
            self.data[column_y] = 1

    @staticmethod
    def convert_dataset_to_ohe(data: pd.DataFrame, categories: list) -> pd.DataFrame:
        labeled = []
        onehot = {}
        for c in categories:
            onehot[c] = OneHotEncoder(handle_unknown="ignore")
            oht = onehot[c].fit_transform(data[[c]])

            labeled.append(
                pd.DataFrame(oht.toarray(), columns=onehot[c].get_feature_names_out()).reset_index(
                    drop=True
                )
            )

        return pd.concat([data.drop(categories, axis=1), *labeled], axis=1)

    def generate_neg_simulated_data(self, geodata: str = "group") -> None:
        events2 = self.data.copy()
        events2["happen"] = 1

        if not isinstance(events2["pub_millis"].iloc[0], np.integer):
            events2["pub_millis"] = events2["pub_millis"].astype(np.int64, errors="ignore") / 1_000_000

        step = np.int64(60_000 * 5)

        min_tms = events2["pub_millis"].min()
        max_tms = events2["pub_millis"].max()

        intervals = np.arange(min_tms, max_tms + step, step)

        events2["interval"] = (events2["pub_millis"].to_numpy() // step) * step

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

        merged["pub_millis"] = merged["interval"]

        result = merged[["pub_millis", geodata, "type", "happen"]]

        if not isinstance(self.data["pub_millis"].iloc[0], np.integer):
            result["pub_millis"] = pd.to_datetime((result["pub_millis"] * 1_000_000), unit="ns")
        else:
            result["pub_millis"] = result["pub_millis"]

        happen_data = result.get("happen", {})
        result.drop(columns=["happen"], inplace = True)

        alerts = utils.generate_aggregate_data(result)

        self.total_events = alerts.data
        self.total_events["happen"] = happen_data

    def balance_day_type(self) -> None:
        # Balance no events and events, for weekend and workdays
        if (
            self.total_events is None
            or "day_type" not in self.data.columns
            or "day_type" not in self.total_events.columns
        ):
            raise ValueError("There are not day type data in dataset")

        logger.info(self.data.day_type.value_counts())
        logger.info(self.total_events.day_type.value_counts())

        days0 = self.data[(self.data.day_type == "f") | (self.data.day_type == 0)]
        days1 = self.data[(self.data.day_type == "s") | (self.data.day_type == 1)]
        q_days0 = days0.shape[0]
        q_days1 = days1.shape[0]

        diff = q_days1 - q_days0

        self.data = cast(
            pd.DataFrame,
            pd.concat([days1, days0.sample(q_days0 + diff, replace=True, random_state=42)]),
        )

        days0 = self.total_events[(self.total_events.day_type == "f") | (self.total_events.day_type == 0)]
        days1 = self.total_events[(self.total_events.day_type == "s") | (self.total_events.day_type == 1)]

        self.total_events = cast(
            pd.DataFrame,
            pd.concat([
                days1.sample(q_days1, replace=True, random_state=42),
                days0.sample(q_days0 + diff, replace=True, random_state=42),
            ]),
        )

    def clean(self, columns_x: List, column_y: str) -> None:
        self.columns_x = columns_x
        self.column_y = column_y

        # Clean the data
        self.data = self.data.loc[:, columns_x + [column_y]]
        if self.total_events is not None:
            self.total_events = self.total_events.loc[:, columns_x + [column_y]]

    def prepare(self, no_features=None) -> None:
        if self.data is None:
            raise ValueError("Not have data for model generation")

        total_events = (
            pd.concat([self.data, self.total_events])
            if self.total_events is not None
            else self.data.copy()
        )

        if "day_type" in total_events.columns:
            total_events["day_type"] = total_events["day_type"].replace({"f": 0, "s": 1})

        if self.ohe:
            labeled = []
            self.onehot = {}

            if self.categories is None:
                self.categories = list(self.data.columns)
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
            if self.categories is None:
                self.categories = list(self.data.columns)
            for c in self.categories:
                self.hasher[c] = FeatureHasher(n_features=no_features, input_type="string")

                # Remove null values and convert to string
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

    def train(self) -> None:
        if self.data_labeled is None:
            self.prepare()

        if self.data_labeled is None:
            return

        self.x = self.data_labeled.drop(self.column_y, axis=1)
        self.y = self.data_labeled.loc[:, self.column_y]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

        self.model.fit(self.x_train, self.y_train)

    def cross_validation(self) -> List:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data labeled is empty, did you prepare the model?"
        x = self.data_labeled.drop(self.column_y, axis=1)
        y = self.data_labeled[self.column_y]

        scores = cross_val_score(self.model, x, y, cv=10)

        return scores

    def predict(self, data) -> List:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data is not labeled, did you prepare the model?"
        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict(data)

    def predict_proba(self, data) -> None:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data is not labeled, did you prepare the model?"
        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict_proba(data)

    def select_best_params(self) -> List:
        if self.data_labeled is None:
            self.train()

        return self.model.best_params_

    def metrics(self) -> Dict:
        y_pred = self.model.predict(self.x_test)

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
        }

        return metrics

    def confusion_matrix(self) -> List:
        if self.y_train is None:
            return []
        return confusion_matrix(
            self.y_test,
            self.model.predict(self.x_test),
            labels=self.y_train.unique(),
        )

    def encode(self, data, category) -> np.ndarray:
        if self.data_labeled is None:
            self.train()

        if self.ohe and self.onehot:
            return (
                self.onehot[category].transform(pd.DataFrame([data], columns=[category])).toarray()
            )
        elif self.hash and self.hasher:
            return self.hasher[category].transform([[s] for s in data]).toarray()
        return np.array([])

    @mlflow_logger
    def log_model_params(self, **params) -> None:
        if self.x_train is None:
            return

        signature = infer_signature(self.x_train, self.model.predict(self.x_train))

        if isinstance(self.model, XGBClassifier):
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
        mlflow.log_metric("cv_score_avg", np.average(scores).astype(float))
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

    @staticmethod
    def get_last_model(model_name: str) -> int:
        client = mlflow.MlflowClient()

        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(model_versions, key=lambda v: int(v.version))

        return int(latest_version.version)
