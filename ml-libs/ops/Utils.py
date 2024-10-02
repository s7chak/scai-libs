import math

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import re
import time
import ops.MLConfig as MLConfig


def time_this(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to run '{func.__name__}': {elapsed_time:.4f} seconds")
        return result

    return wrapper

class Util:
    def __init__(self):
        self.scaler = None

    def check_split(self, split):
        if isinstance(split, str):
            fraction_pattern = re.compile(r"^\d*\.?\d+$")
            if fraction_pattern.match(split):
                return float(split)
            try:
                split_date = datetime.strptime(split, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return split_date
            except ValueError:
                raise ValueError("Invalid split value: must be a fraction or a date string.")
        raise TypeError("Split value must be a string.")

    def plot_lines(self, df, label=''):
        traces = []
        for c in df.columns:
            traces.append(go.Scatter(x=df.index, y=df[c].values, name=str(c) + label))
        return traces

    def plot_bars(self, df):
        traces = []
        df.sort_values(df.columns[0], inplace=True)
        for c in df.columns:
            traces.append(go.Bar(x=df.index, y=df[c].values, name=c))
        return traces

    def plot_comparison(self, actual, predicted_baseline, predicted_best, trials=[], stock=''):
        plt.plot(actual, color='black', label='Actual')
        plt.plot(predicted_best, color='green', label='Best Model')
        plt.plot(predicted_baseline, color='blue', label='Base Model')
        plt.title('Best vs Baseline model')
        plt.xlabel('Time')
        plt.ylabel(stock + ' Stock Price')
        if len(trials) > 0:
            for i, pred in enumerate(trials):
                plt.plot(pred, label='Trial:' + str(i))
        plt.legend()
        plt.show()


class Metrics:

    def calculate_metric(self, metricname, true, pred):
        metric = 999
        if metricname == 'rmse':
            metric = math.sqrt(mean_squared_error(true, pred))
        if metricname == 'mse':
            metric = mean_squared_error(true, pred)
        if metricname == 'mae':
            metric = mean_absolute_error(true, pred)
        if metricname == 'mape':
            metric = mean_absolute_percentage_error(true, pred)
        if metricname == 'r2':
            metric = r2_score(true, pred)
        return round(metric, MLConfig.metric_significant_digits)
