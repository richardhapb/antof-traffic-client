from sklearn.base import BaseEstimator
from analytics.grouper import Grouper


class Alerts:
    def __init__(self, data: Grouper | None = None):
        self.data = data


class TimeRange:
    def __init__(self, init_time: int, end_time: int, selected_time: int):
        self.init_time = init_time
        self.end_time = end_time
        self.selected_time = selected_time


class Model:
    def __init__(self, model_obj: BaseEstimator | None = None, last_model: int = 0):
        self.model = model_obj
        self.last_model = last_model
