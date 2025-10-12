import os
from functools import wraps
from typing import cast
from collections.abc import Callable
import uuid

import numpy as np
import pandas as pd
import geopandas as gpd
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
from utils.utils import join_coords, logger


MODEL_NAME = "XGBClassifier"


class TrainError(Exception):
    """Error type for training process"""

    def __init__(self, msg: str):
        super().__init__(f"Training error: {msg}")


def init_mlflow() -> None:
    """Init MLFlow service"""

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
    mlflow.set_tracking_uri(uri=mlflow_uri)
    mlflow.set_experiment("AntofTraffic")


def mlflow_logger(func: Callable):  # noqa: ANN201
    """Wrap for log data of a MLFlow run"""

    @wraps(func)
    def mlflow_wrapper(*args, **kwargs):  # noqa: ANN202
        with mlflow.start_run():
            return func(*args, **kwargs)

    return mlflow_wrapper


class ML:
    """
    A machine learning wrapper class for training and evaluating XGBoost classification models.

    This class handles data preparation, model training, evaluation and MLflow logging for
    XGBoost classification tasks with support for feature encoding.

    Attributes:
        data (GeoDataFrame | pd.DataFrame): Input training data
        model (XGBClassifier): XGBoost classifier model instance
        column_y (str | None): Target column name
        categories (List | None): Categorical columns to encode
        ohe (bool): Whether to use one-hot encoding
        hash (bool): Whether to use feature hashing
        hasher (Dict[str, FeatureHasher] | None): Feature hashers by column
        data_labeled (GeoDataFrame | pd.DataFrame | None): Encoded training data
        x_train (pd.DataFrame | None): Training features
        x_test (pd.DataFrame | None): Test features
        y_train (pd.DataFrame | None): Training labels
        y_test (pd.DataFrame | None): Test labels
        onehot (Dict[str, OneHotEncoder] | None): One-hot encoders by column
        oe (Dict[str, OrdinalEncoder] | None): Ordinal encoders by column
        total_events (pd.DataFrame | GeoDataFrame | None): Combined event data
        x (np.ndarray | None): Feature matrix
        y (np.ndarray | None): Label vector

    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        data: GeoDataFrame | pd.DataFrame,
        model: XGBClassifier,
        column_y: str | None = None,
        categories: list | None = None,
        ohe: bool = False,
        hash: bool = False,
    ):
        """
        Initialize the ML wrapper.

        Args:
            data: Input training data
            model: XGBoost classifier model instance
            column_y: Name of target column
            categories: List of categorical columns to encode
            ohe: Whether to use one-hot encoding
            hash: Whether to use feature hashing

        """
        self.data: GeoDataFrame | pd.DataFrame = data
        self.model: XGBClassifier = cast("XGBClassifier", clone(model))
        self.columns_x: list = [column for column in self.data.columns if column != column_y]
        self.column_y: str | None = column_y
        self.categories: list | None = categories
        self.ohe: bool = ohe
        self.hash: bool = hash
        self.hasher: dict[str, FeatureHasher] | None = None
        self.data_labeled: GeoDataFrame | pd.DataFrame | None = None
        self.x_train: list | pd.DataFrame | None = None
        self.x_test: list | pd.DataFrame | None = None
        self.y_train: list | pd.Series | None = None
        self.y_test: list | pd.DataFrame | None = None
        self.onehot: dict[str, OneHotEncoder] | None = None
        self.oe: dict[str, OrdinalEncoder] | None = None  # Ordinal Encoder
        self.total_events: pd.DataFrame | GeoDataFrame | None = None
        self.x: np.ndarray | gpd.GeoDataFrame | pd.DataFrame | None = None
        self.y: np.ndarray | pd.Series | pd.DataFrame | None = None

        if self.data is not None:
            self.data[column_y] = 1

    def check_labeled_data(self) -> None:
        """Check if the data is labeled and encoded correctly, and raise a `TrainError` if it is not"""

        if self.data_labeled is None:
            msg = "Data is not labeled, did you prepare the model?"
            raise TrainError(msg)

    @property
    def labeled_data(self) -> pd.DataFrame | GeoDataFrame:
        """Return self.data_labeled ensuring it is not None."""
        if self.data_labeled is None:
            self.train()
        self.check_labeled_data()
        return self.data_labeled  # type: ignore

    @staticmethod
    def convert_dataset_to_ohe(data: pd.DataFrame, categories: list) -> pd.DataFrame:
        """
        Convert categorical columns to one-hot encoded features.

        Args:
            data: Input dataframe
            categories: List of categorical column names to encode

        Returns:
            DataFrame with one-hot encoded features

        """

        labeled = []
        onehot = {}
        for c in categories:
            onehot[c] = OneHotEncoder(handle_unknown="ignore")
            oht = onehot[c].fit_transform(data[[c]])

            labeled.append(
                pd.DataFrame(oht.toarray(), columns=onehot[c].get_feature_names_out()).reset_index(drop=True)
            )

        return pd.concat([data.drop(categories, axis=1), *labeled], axis=1)

    @staticmethod
    def generate_neg_simulated_data(data: pd.DataFrame | GeoDataFrame, geodata: str = "group") -> GeoDataFrame:
        """
        Generate negative samples by simulating non-event data points.

        Creates synthetic negative examples by sampling timestamps and locations
        where events did not occur. Uses a 5-minute interval grid and combines
        with location groups and event types.

        Args:
            data: Data to process
            geodata: Column name containing geographical grouping information.
                    Defaults to "group".

        """

        if not isinstance(data["pub_millis"].iloc[0], np.integer):
            data["pub_millis"] = data["pub_millis"].astype(np.int64, errors="ignore") / 1_000_000

        step = np.int64(60_000 * 5)

        min_tms = data["pub_millis"].to_numpy().min()
        max_tms = data["pub_millis"].to_numpy().max()

        intervals = np.arange(min_tms, max_tms + step, step)

        data["interval"] = ((data["pub_millis"].to_numpy() - min_tms) // step) * step + min_tms

        allgroups = data[geodata].unique()
        alltypes = data["type"].unique()

        combinations = pd.MultiIndex.from_product(
            [intervals, allgroups, alltypes], names=["interval", geodata, "type"]
        ).to_frame(index=False)

        event_combinations = data[["uuid", "interval", geodata, "type", "x", "y"]]
        logger.debug(
            "data combinations: %d, simulated combinations: %d. Before resample",
            event_combinations.shape[0],
            combinations.shape[0],
        )

        event_combinations["happen"] = 1
        event_combinations["location"] = join_coords(event_combinations)

        merged = pd.merge(
            combinations,
            event_combinations,
            on=["interval", geodata, "type"],
            how="left",
        )

        data.drop(columns=["interval"], inplace=True)

        merged_pos = merged[merged["happen"] == 1]
        happen_len = merged_pos.shape[0]

        merged_neg = merged[merged["happen"].isna()].sample(
            happen_len, replace=False, random_state=42
        ).reset_index(drop=True)

        full_data = cast("pd.DataFrame", pd.concat([merged_pos, merged_neg]))

        # Create the location list for the non-event samples
        locations = event_combinations.loc[:, "location"].values.tolist()
        mask = full_data["happen"].isna()
        no_n = mask.sum()

        # Repeat and trim the locations list to match the number of non-event samples
        locations *= no_n // len(locations) + 1
        locations = locations[:no_n]

        np.random.shuffle(locations)
        full_data.loc[mask, "location"] = locations

        n_pos = (full_data["happen"] == 1).shape[0]
        n_neg = (full_data["happen"].isna()).shape[0]

        logger.debug("Balance of classes happen:no-happen, %d:%d", n_pos, n_neg)

        full_data["happen"] = full_data["happen"].fillna(0).astype(int)
        full_data["pub_millis"] = full_data["interval"]

        result = full_data[["uuid", "pub_millis", geodata, "type", "happen", "location"]]

        if not isinstance(data["pub_millis"].iloc[0], np.integer):
            result["pub_millis"] = pd.to_datetime((result["pub_millis"]), unit="ms")

        happen_data = result["happen"].to_numpy(copy=False)
        result.drop(columns=["happen"], inplace=True)

        result.fillna({"uuid": uuid.uuid4().hex}, inplace=True)
        result.reset_index(drop=True, inplace=True)

        alerts = utils.generate_aggregate_data(result)

        alerts.data["happen"] = happen_data

        return alerts.data

    def insert_neg_simulated_data(self, geodata: str = "group") -> None:
        self.total_events = self.generate_neg_simulated_data(self.data, geodata)

    def balance_day_type(self) -> None:
        """
        Balance the dataset between weekday and weekend samples.

        Resamples data to ensure equal representation of weekday ("f") and
        weekend ("s") events. Uses random sampling with replacement to match
        the counts of the larger class.

        Raises:
            ValueError: If day_type column is missing from the dataset

        """

        if (
            self.total_events is None
            or "day_type" not in self.data.columns
            or "day_type" not in self.total_events.columns
        ):
            msg = "There are not day type data in dataset"
            raise ValueError(msg)

        logger.info(self.data.day_type.value_counts())
        logger.info(self.total_events.day_type.value_counts())

        days0 = self.data[(self.data.day_type == "f") | (self.data.day_type == 0)]
        days1 = self.data[(self.data.day_type == "s") | (self.data.day_type == 1)]
        q_days0 = days0.shape[0]
        q_days1 = days1.shape[0]

        diff = q_days1 - q_days0

        self.data = cast(
            "pd.DataFrame",
            pd.concat([days1, days0.sample(q_days0 + diff, replace=True, random_state=42)]),
        )

        days0 = self.total_events[(self.total_events.day_type == "f") | (self.total_events.day_type == 0)]
        days1 = self.total_events[(self.total_events.day_type == "s") | (self.total_events.day_type == 1)]

        self.total_events = cast(
            "pd.DataFrame",
            pd.concat([
                days1.sample(q_days1, replace=True, random_state=42),
                days0.sample(q_days0 + diff, replace=True, random_state=42),
            ]),
        )

    def clean(self, columns_x: list, column_y: str) -> None:
        """
        Filter dataset to include only specified feature and target columns.

        Args:
            columns_x: List of feature column names to keep
            column_y: Name of the target column

        Raises:
            ValueError: If required columns don't exist in the dataset

        """

        self.columns_x = columns_x
        self.column_y = column_y

        # Verify required columns exist in the primary dataset
        missing_cols = [col for col in [*columns_x, column_y] if col not in self.data.columns]
        if missing_cols:
            msg = f"Missing required columns in data: {missing_cols}"
            raise ValueError(msg)

        # Clean the primary data
        self.data = self.data.loc[:, [*columns_x, column_y]]

        # Handle total_events if it exists
        if self.total_events is not None:
            missing_cols = [col for col in [*columns_x, column_y] if col not in self.total_events.columns]
            if missing_cols:
                msg = f"Missing required columns in total_events: {missing_cols}"
                raise ValueError(msg)

            self.total_events = self.total_events.loc[:, [*columns_x, column_y]]

    def prepare(self, n_features: int | None = None) -> None:
        """
        Prepare data for model training by applying feature encoding.

        Handles data preparation including:
        - Combining event and non-event data
        - Converting day type indicators
        - Applying one-hot encoding or feature hashing if specified

        Args:
            n_features: Number of features for feature hashing. Only used if
                        hash=True. Defaults to 5 if not specified.

        Raises:
            ValueError: If no data is available for model generation

        """

        if self.data is None:
            msg = "Not have data for model generation"
            raise ValueError(msg)

        total_events = pd.concat([self.data, self.total_events]) if self.total_events is not None else self.data.copy()

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
                    pd.DataFrame(oht.toarray(), columns=self.onehot[c].get_feature_names_out()).reset_index(drop=True)
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
            n_features = n_features if n_features else 5
            if self.categories is None:
                self.categories = list(self.data.columns)
            for c in self.categories:
                self.hasher[c] = FeatureHasher(n_features=n_features, input_type="string")

                # Remove null values and convert to string
                filtered_events = [str(s) for s in total_events[c] if pd.notnull(s)]
                hashed = self.hasher[c].fit_transform([[s] for s in filtered_events])

                labeled.append(
                    pd.DataFrame(
                        hashed.toarray(),
                        columns=[f"{c}_{i}" for i in range(n_features)],
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
        """
        Train the XGBoost model on the prepared data.

        Performs the following steps:
            1. Prepares data if not already done
            2. Splits features and target
            3. Creates train/test split (80/20)
            4. Fits the XGBoost model
        """

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

    def cross_validation(self) -> list:
        """
        Perform 10-fold cross validation on the model.

        Returns:
            List of scores from each fold of cross validation

        Raises:
            TrainError: If data has not been prepared

        """

        if self.data_labeled is None:
            self.train()

        x = self.labeled_data.drop(self.column_y, axis=1)
        y = self.labeled_data[self.column_y]

        return cross_val_score(self.model, x, y, cv=10)

    def predict(self, data: pd.DataFrame | gpd.GeoDataFrame) -> list:
        """
        Make predictions on new data using the trained model.

        Args:
            data: DataFrame containing features for prediction

        Returns:
            List of predicted class labels

        Raises:
            ValueError: If model has not been trained

        """

        if self.data_labeled is None:
            self.train()

        if self.data_labeled is None:
            msg = "Data is not labeled, did you prepare the model?"
            raise TrainError(msg)

        data = data[[e for e in list(self.data_labeled.columns) if e != self.column_y]]

        return self.model.predict(data)

    def predict_proba(self, data: pd.DataFrame | gpd.GeoDataFrame) -> None:
        """
        Get prediction probabilities for each class on new data.

        Args:
            data: DataFrame containing features for prediction

        Returns:
            Array of predicted probabilities for each class

        Raises:
            TrainError: If model has not been trained

        """

        if self.data_labeled is None:
            self.train()

        data = data[[e for e in list(self.labeled_data.columns) if e != self.column_y]]

        return self.model.predict_proba(data)

    def select_best_params(self) -> list:
        """
        Get the best hyperparameters from model tuning.

        Returns:
            Dictionary of best parameter values found during tuning

        """

        if self.data_labeled is None:
            self.train()

        return self.model.best_params_

    def metrics(self) -> dict:
        """
        Calculate performance metrics for the model.

        Returns:
            Dictionary containing:
                - accuracy: Accuracy score
                - f1: F1 score
                - precision: Precision score
                - recall: Recall score

        """

        y_pred = self.model.predict(self.x_test)

        return {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
        }

    def confusion_matrix(self) -> np.ndarray:
        """
        Generate confusion matrix for model predictions.

        Returns:
            2D array representing the confusion matrix, empty list if model
            hasn't been trained

        """

        if self.y_train is None:
            return np.ndarray([])
        return confusion_matrix(
            self.y_test,
            self.model.predict(self.x_test),
            labels=self.y_train.unique() if isinstance(self.y_train, pd.Series) else self.y_train,
        )

    def encode(self, data: pd.DataFrame | gpd.GeoDataFrame, category: str) -> np.ndarray:
        """
        Encode new categorical data using fitted encoders.

        Args:
            data: Data to encode
            category: Name of the categorical column

        Returns:
            numpy.ndarray: Encoded features using either one-hot encoding,
                          feature hashing, or empty array if no encoding set

        """

        if self.data_labeled is None:
            self.train()

        if self.ohe and self.onehot:
            return self.onehot[category].transform(pd.DataFrame([data], columns=[category])).toarray()

        if self.hash and self.hasher:
            return self.hasher[category].transform([[s] for s in data]).toarray()

        return np.array([])

    @mlflow_logger
    def log_model_params(self, **params) -> None:
        """
        Log model parameters, metrics and artifacts to MLflow.

        Logs:
            - Model artifact
            - Model parameters
            - Cross validation scores
            - Confusion matrix
            - Performance metrics (accuracy, f1, precision, recall)
            - Additional custom parameters

        Args:
            **params: Additional parameters to log

        """

        if self.x_train is None:
            return

        signature = infer_signature(self.x_train, self.model.predict(self.x_train))

        if isinstance(self.model, XGBClassifier) and isinstance(self.x_train, (pd.DataFrame, gpd.GeoDataFrame)):
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

    def prepare_train(self) -> None:
        """Prepare data and train model in one step."""

        self.prepare()
        self.train()

    def ordinal_encoder(self, categories: list) -> None:
        """
        Apply ordinal encoding to specified categorical columns.

        Args:
            categories: List of categorical column names to encode

        Raises:
            ValueError: If data is not loaded or specified category
                       doesn't exist in data

        """

        if self.data is None:
            msg = "Data is not loaded"
            raise ValueError(msg)
        self.oe = {}
        for cat in categories:
            if cat not in self.data.columns:
                raise ValueError(cat + " is not in data")
            self.oe[cat] = OrdinalEncoder()
            self.data[cat] = self.oe[cat].fit_transform(self.data[[cat]])

    @staticmethod
    def get_last_model(model_name: str) -> int:
        """
        Get the version number of the latest model in MLflow.

        Args:
            model_name: Name of the registered model

        Returns:
            Integer version number of the latest model

        """

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
        mlflow.set_tracking_uri(uri=mlflow_uri)

        model_versions = mlflow.search_model_versions(None, f"name='{model_name}'")
        latest_version = max(model_versions, key=lambda v: int(v.version))

        return int(latest_version.version)
