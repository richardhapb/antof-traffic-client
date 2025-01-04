from analytics.ml import ML, init_mlflow
from analytics.grouper import Grouper
from utils import utils
from xgboost import XGBClassifier
from datetime import datetime
import pytz

def train():
    init_mlflow()

    print("Extracting data from database")

    COLS = ["type", "geometry", "hour", "day_type", "week_day", "day"]

    alerts_events = utils.load_data("alerts")
    alerts_gdf = alerts_events.to_gdf()
    alerts_gdf = utils.extract_event(alerts_gdf,
                                     ["ACCIDENT", "JAM", "HAZARD", "ROAD_CLOSED"],
                                     COLS
                                     )
    alerts = Grouper(alerts_gdf)
    alerts.group((10, 20))

    print("Data extracted and transformed")

    model = XGBClassifier(
        learning_rate=0.1,
        random_state=42,
        n_estimators=80,
        max_depth=20,
        gamma=0.8,
        colsample_bytree=0.7,
    )

    x_vars = ["group", "hour", "day_type", "type", "week_day", "day"]
    y = "happen"
    categories = ["type"]

    ml = ML(alerts.data, model, y, categories, True)

    print("Training the model")

    ml.generate_neg_simulated_data(x_vars)
    ml.clean(x_vars, y)
    ml.prepare_train()

    print("Model trained")


    ml.log_model_params(
        hash_encode=ml.hash,
        ohe=ml.ohe,
        sample=ml.data.shape,
        ordinal_encoder=False,
        sample_no_events=ml.no_events.shape if ml.no_events is not None else [],
        geodata="group",
        categories=categories,
    )

    print("Model registered")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error generating model, time {datetime.now(tz=pytz.timezone('America/Santiago'))}")
        print(f"Error: {e}")
