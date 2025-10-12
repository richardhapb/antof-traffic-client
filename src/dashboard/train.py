from datetime import datetime

import pytz
import requests
from xgboost import XGBClassifier

from analytics.ml import ML, init_mlflow
from utils import utils
from utils.utils import TZ, logger, ALERTS_BEGIN_TIMESTAMP


def train() -> bool:
    """Trigger the model trainning"""

    init_mlflow()

    logger.info("Extracting data from server")

    now_millis = int(datetime.now(pytz.UTC).timestamp()) * 1000

    try:
        alerts = utils.get_data(ALERTS_BEGIN_TIMESTAMP, now_millis)  # Get all data
        alerts.filter_by_group_time(120, True)
    except requests.ConnectionError:
        logger.exception("Error retrieving data from server")
        return False

    logger.info("Data found: %i", alerts.data.shape[0])

    # Best params tested with GridSearch
    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        colsample_bytree=0.8,
        gamma=0.8,
        learning_rate=0.1,
        max_depth=20,
        min_child_weight=1,
        subsample=0.8,
    )

    x_vars = ["group", "hour", "day_type", "type", "week_day", "day"]
    y = "happen"
    categories = ["type"]

    ml = ML(alerts.data, model, y, categories, True)

    logger.info("Training the model")

    ml.insert_neg_simulated_data()
    ml.clean(x_vars, y)
    ml.prepare_train()

    logger.info("Model trained")

    ml.log_model_params(
        hash_encode=ml.hash,
        ohe=ml.ohe,
        sample=ml.data.shape,
        ordinal_encoder=False,
        sample_no_events=ml.total_events.shape if ml.total_events is not None else [],
        geodata="group",
        categories=categories,
    )

    logger.info("Model registered")

    return True


if __name__ == "__main__":
    try:
        train()
    except Exception:
        logger.error("Error generating model, time %s", datetime.now(tz=pytz.timezone(TZ)))
        logger.exception("Error")
