import mlflow

# from analytics.ml import ML
from dashboard.models import Model
from analytics.ml import MODEL_NAME


def load_model(model_obj: Model):
    model_name = MODEL_NAME
    model_obj.last_model = 6 # ML.get_last_model(model_name)
    model_obj.model = mlflow.sklearn.load_model(
        f"models:/{model_name}/{model_obj.last_model}"
    )
    print(f"Model {model_name} version {model_obj.last_model} successfully loaded")
