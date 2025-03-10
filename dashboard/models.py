from xgboost import XGBClassifier
from analytics.grouper import Grouper


class Alerts:
    def __init__(self, data: Grouper | None = None):
        self.data = data


class TimeRange:
    def __init__(self, init_time: int, end_time: int, selected_time: int | None = None):
        if selected_time is None:
            selected_time = end_time

        self.init_time = init_time
        self.end_time = end_time
        self.selected_time = selected_time


class Model:
    def __init__(self, model_obj: XGBClassifier = None, last_model: int = 0):
        self.model = model_obj
        self.last_model = last_model
