from typing import Dict, List, cast
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
from sklearn.base import clone, ClassifierMixin
import pandas as pd
from analytics.grouper import Grouper
from utils import utils
import matplotlib.pyplot as plt
import contextily as cx
import mlflow
from mlflow.models import infer_signature
from functools import wraps
import numpy as np
from geopandas import GeoDataFrame
from matplotlib.figure import Figure

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
            self,
            data:GeoDataFrame | pd.DataFrame,
            model:ClassifierMixin,
            column_y:str | None = None,
            categories: List | None = None,
            ohe: bool=False,
            hash:bool=False
    ):
        self.data:GeoDataFrame | pd.DataFrame = data
        self.model: ClassifierMixin = cast(ClassifierMixin, clone(model)) 
        self.columns_x : List = [column for column in self.data.columns if column != column_y]
        self.column_y : str | None = column_y
        self.categories: List | None = categories
        self.ohe: bool = ohe
        self.hash: bool = hash
        self.hasher: Dict[ str, FeatureHasher ] | None = None
        self.data_labeled: GeoDataFrame | pd.DataFrame | None = None
        self.x_train:pd.DataFrame | None = None
        self.x_test:pd.DataFrame | None = None
        self.y_train:pd.DataFrame | None = None
        self.y_test:pd.DataFrame | None = None
        self.onehot: Dict[ str, OneHotEncoder ] | None = None
        self.oe: Dict[ str, OrdinalEncoder ] | None = None # Ordinal Encoder
        self.no_events: pd.DataFrame | GeoDataFrame | None = None
        self.x : np.ndarray | None = None
        self.y: np.ndarray | None = None

        if self.data is not None:
            self.data[column_y] = 1

    @staticmethod
    def convert_dataset_to_ohe(data: pd.DataFrame, categories: list)->pd.DataFrame:
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

    def generate_neg_simulated_data(self, extra_cols: List, geodata: str = "group")->None:
        events2 = self.data.copy()
        events2["happen"] = 1

        if not isinstance(events2["inicio"].iloc[0], np.integer):
            events2["inicio"] = events2["inicio"].astype(np.int64, errors="ignore") / 1_000_000

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

        if not isinstance(self.data["inicio"].iloc[0], np.integer):
            result["inicio"] = pd.to_datetime((result["inicio"] * 1_000_000), unit="ns")
        else:
            result["inicio"] = result["inicio"]

        self.no_events = pd.DataFrame(result)
        self.no_events = utils.extract_event(
            self.no_events, CONCEPTS, extra_cols + ["happen"]
        )

    def balance_day_type(self)->None:
        # Balance no events and events, for weekend and workdays
        if (
            self.no_events is None
            or
            "day_type" not in self.data.columns
            or "day_type" not in self.no_events.columns
        ):
            raise ValueError("There are not day type data in dataset")

        print(self.data.day_type.value_counts())
        print(self.no_events.day_type.value_counts())

        days0 = self.data[(self.data.day_type == "f") | (self.data.day_type == 0)]
        days1 = self.data[(self.data.day_type == "s") | (self.data.day_type == 1)]
        q_days0 = days0.shape[0]
        q_days1 = days1.shape[0]

        diff = q_days1 - q_days0

        self.data = pd.concat(
            [days1, days0.sample(q_days0 + diff, replace=True, random_state=42)]
            ) # type: ignore

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
            ) # type: ignore

    def clean(self, columns_x: List, column_y:str)->None:
        self.columns_x = columns_x
        self.column_y = column_y

        # Clean the data
        self.data = self.data.loc[:, columns_x + [column_y]]
        if self.no_events is not None:
            self.no_events = self.no_events.loc[:, columns_x + [column_y]]

    def prepare(self, no_features=None)->None:
        if self.data is None:
            raise ValueError("Not have data for model generation")

        total_events = (
            pd.concat([self.data, self.no_events])
            if self.no_events is not None
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
                self.hasher[c] = FeatureHasher(
                    n_features=no_features, input_type="string"
                )
                # Eliminar valores nulos y convertir todos los valores a strings
                filtered_events = [str(s) for s in total_events[c] if pd.notnull(s)]
                hashed = self.hasher[c].fit_transform([[s] for s in filtered_events])

                labeled.append(
                    pd.DataFrame(
                        hashed.toarray(),
                        columns=[f"{c}_{i}" for i in range(no_features)], # type: ignore
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

    def train(self)->None:
        if self.data_labeled is None:
            self.prepare()
        self.x = self.data_labeled.drop(self.column_y, axis=1) # type: ignore
        self.y = self.data_labeled.loc[:, self.column_y] # type: ignore

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( # type: ignore
            self.x, self.y, test_size=0.2, random_state=42
            ) 

        self.model.fit(self.x_train, self.y_train) # type: ignore

    def cross_validation(self)->List:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data labeled is empty, did you prepare the model?"
        x = self.data_labeled.drop(self.column_y, axis=1)
        y = self.data_labeled[self.column_y]

        scores = cross_val_score(self.model, x, y, cv=10)

        return scores

    def predict(self, data)->List:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data is not labeled, did you prepare the model?"
        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict(data) # type: ignore

    def predict_proba(self, data)->None:
        if self.data_labeled is None:
            self.train()

        assert self.data_labeled is not None, "Data is not labeled, did you prepare the model?"
        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict_proba(data) # type: ignore

    def select_best_params(self)->List:
        if self.data_labeled is None:
            self.train()

        return self.model.best_params_ # type: ignore

    def metrics(self)->Dict:
        y_pred = self.model.predict(self.x_test) # type: ignore

        metrics = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
        }

        return metrics

    def confusion_matrix(self)->List:
        if self.y_train is None:
            return []
        return confusion_matrix(
            self.y_test,
            self.model.predict(self.x_test), # type: ignore
            labels=self.y_train.unique(),
        )

    def encode(self, data, category)->np.ndarray:
        if self.data_labeled is None:
            self.train()

        if self.ohe and self.onehot:
            return (
                self.onehot[category]
                .transform(pd.DataFrame([data], columns=[category])) # type: ignore
                .toarray()
            )
        elif self.hash and self.hasher:
            return self.hasher[category].transform([[s] for s in data]).toarray()
        return np.array([])

    def plot_by_quad(self, grouper: Grouper, obj: pd.DataFrame)->Figure:
        obj = obj.copy()
        if self.data_labeled is None:
            self.train()
        fig, ax = plt.subplots()

        fig.set_size_inches((4.5, 9.5))
        xc, yc = grouper.get_center_points()
        i, j = 0, 0
        between_x = xc[0][1] - xc[0][0]
        between_y = yc[1][0] - yc[0][0]

        if self.categories is None:
            self.categories = list(self.data.columns)

        for xp in xc[0]:
            for yp in yc.T[0]:
                quad = grouper.calc_quadrant(i, j)
                xf = xp - between_x / 2
                yf = yp - between_y / 2
                if self.ohe and "group" in self.categories:
                    obj["group_" + str(quad)] = 1
                else:
                    obj["group"] = quad
                pred = self.predict_proba(obj)
                pred = pred[0][1] if pred else 0

                ax.text(xp - 250, yp - 150, str(round(pred, 1)), fontsize=6, alpha=0.8)
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
            source=cx.providers.OpenStreetMap.Mapnik, # type: ignore
        )

        ax.set_title("Accidentes por cuadrante")
        ax.set_ylabel("Latitud")
        ax.set_xlabel("Longitud")
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig

    @mlflow_logger
    def log_model_params(self, **params)->None:

        if self.x_train is None:
            return

        signature = infer_signature(self.x_train, self.model.predict(self.x_train)) # type: ignore

        if isinstance(self.model, ClassifierMixin):
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="waze_data",
                signature=signature,
                input_example=self.x_train.head(), 
                registered_model_name=type(self.model).__name__,
            )

        model_params = self.model.get_params() # type: ignore
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
    def get_last_model(model_name: str)->int:
        client = mlflow.tracking.MlflowClient()
        
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(model_versions, key=lambda v: int(v.version))

        return int(latest_version.version)
