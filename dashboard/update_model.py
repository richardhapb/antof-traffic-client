import mlflow

# from analytics.ml import ML
from dashboard.models import Model
from analytics.ml import MODEL_NAME

from utils.utils import logger


def load_model(model_obj: Model) -> None:
    """Get last model from MLFlow and load it"""

    model_name = MODEL_NAME
    # TODO: Update this when model number 8 has been trained
    model_obj.last_model = 6  # ML.get_last_model(model_name)
    model_obj.model = mlflow.sklearn.load_model(
        f"models:/{model_name}/{model_obj.last_model}"
    )
    logger.info("Model %s version %s successfully loaded", model_name, model_obj.last_model)
