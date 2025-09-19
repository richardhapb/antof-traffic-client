import mlflow

from analytics.ml import ML
from dashboard.models import Model
from analytics.ml import MODEL_NAME

from utils.utils import logger


def load_model(model_obj: Model) -> None:
    """Get last model from MLFlow and load it"""

    last_version = ML.get_last_model(MODEL_NAME)
    model_obj.model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{last_version}"
    )
    model_obj.last_model = last_version
    logger.info("Model %s version %s successfully loaded", MODEL_NAME, model_obj.last_model)
