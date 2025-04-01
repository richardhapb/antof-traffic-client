from dataclasses import dataclass

from xgboost import XGBClassifier


@dataclass
class TimeRange:
    """Structure for handling the date range"""

    def __init__(self, init_time: int, end_time: int):

        self.init_time = init_time
        self.end_time = end_time


@dataclass
class Model:
    """Structure for managing the trained model"""

    def __init__(self, model_obj: XGBClassifier = None, last_model: int = 0):
        self.model = model_obj
        self.last_model = last_model
