import re

import matplotlib.pyplot as plt
import ops.MLConfig as MLConfig
import optuna
import pandas as pd
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from lightgbm import LGBMRegressor
from ops.Utils import Metrics, Util
from optuna.samplers import TPESampler, RandomSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from tensorflow.keras.layers import Dropout
# from tsmixer import TSMixerModel
from xgboost import XGBRegressor
from ops.Data import Preprocess as Preprocess

class BaseModel:
    def __init__(self, model, m_type=None, name=None):
        self.model = model
        self.m_type = m_type  # statsmodel, sklearn, deeplearning
        self.name = name or type(model).__name__
        self.params = {}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self):
        return self.params

class NNModel(BaseModel):
    def fit(self):
        pass

class ModelFactory:
    @staticmethod
    def get_model(model_type, **params):
        if model_type == 'LinearReg':
            model = LinearRegression()
            return BaseModel(model, m_type=MLConfig.sklearn)

        elif model_type == 'XGBoostRegressor':
            model = XGBRegressor()
            return BaseModel(model, m_type=MLConfig.sklearn)

        elif model_type == 'ARIMA':
            p = params.get('p', 1)
            d = params.get('d', 1)
            q = params.get('q', 1)
            trend = params.get('trend', 'c')
            model = ARIMA(order=(p, d, q), trend=trend)
            return BaseModel(model, m_type=MLConfig.statsmodel)

        elif model_type == 'VARMAX':
            order = params.get('order', (1, 1))
            model = VARMAX(order=order)
            return BaseModel(model, m_type=MLConfig.statsmodel)

        elif model_type == 'DNN':
            model = Sequential([
                Dense(params.get('units', 64), input_shape=(params.get('input_shape', 1),), activation='relu'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=params.get('optimizer', 'adam'), loss='mse')
            return NNModel(model, m_type=MLConfig.dl)

        elif model_type == 'LSTM':
            model = Sequential([
                LSTM(params.get('units', 64), input_shape=(params.get('timesteps', 1), params.get('features', 1)), activation='tanh'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=params.get('optimizer', 'adam'), loss='mse')
            return NNModel(model, m_type=MLConfig.dl)

        elif model_type == 'GRU':
            model = Sequential([
                GRU(params.get('units', 64), input_shape=(params.get('timesteps', 1), params.get('features', 1)), activation='tanh'),
                Dense(1, activation='linear')
            ])
            model.compile(optimizer=params.get('optimizer', 'adam'), loss='mse')
            return NNModel(model, m_type=MLConfig.dl)

        # elif model_type == 'NBeats':
        #     model = NBeats.from_dataset(params['dataset'])
        #     return BaseModel(model, m_type=MLConfig.dl)
        #
        # elif model_type == 'TFT':
        #     model = TemporalFusionTransformer.from_dataset(params['dataset'])
        #     return BaseModel(model, m_type=MLConfig.dl)

        # elif model_type == 'TSMixer':
        #     model = TSMixerModel(params.get('input_size', 64), params.get('output_size', 1), params.get('hidden_size', 64))
        #     return BaseModel(model, m_type='deeplearning')
        else:
            raise ValueError(f"Model type {model_type} not supported.")


class ModelOptimizer:

    def __init__(self, X, y, sampler='TPE', n_trials=100):
        self.X = X
        self.y = y
        self.sampler = TPESampler() if sampler == 'TPE' else RandomSampler()
        self.n_trials = n_trials

    def optimize_models(self, models):
        results = {}

        # Loop over each model and optimize its hyperparameters
        for model in models:
            study = optuna.create_study(sampler=self.sampler, direction='minimize')
            study.optimize(lambda trial: self.objective(trial, model), n_trials=self.n_trials)

            # Store the optimized model and best parameters
            results[model.name] = {
                'best_model': model,
                'best_params': study.best_params,
                'best_value': study.best_value
            }

        return results


class Model:
    def __init__(self, model):
        self.model = model
        n_ = type(model).__name__
        t_ = str(type(model))
        self.name = n_ if 'Sequential' not in n_ else re.search(r"\'(.*?)\'", str(type(model.layers[0]))).group(1)
        self.type = MLConfig.dl if 'Sequential' in t_ else MLConfig.statsmodel if 'statsmodel' in t_ else MLConfig.sklearn


g_epochs, g_batch_size = 2, 50

class TimeSeriesModeler:
    def __init__(self):
        self.x_val = None
        self.x_train = None
        self.y_val = None
        self.y_train = None
        self.data = None
        self.util, self.model_objects = Util(), []
        self.preprocess = Preprocess()
        self.metric_operator = Metrics()

    def data_preprocessing(self, data, split, target, inputs, transform_dict):
        year_train = str(int(split) - 1)
        self.data = data.set_index('Date')
        train, val = self.data[:year_train], self.data[split:]
        transformed_train = self.preprocess.transform(train, transform_dict)
        transformed_val = self.preprocess.transform(val, transform_dict)
        imputed_train = self.preprocess.impute(transformed_train, transform_dict)
        imputed_val = self.preprocess.impute(transformed_val, transform_dict)
        self.y_train = imputed_train[[target]]
        self.y_val = imputed_val[[target]]
        self.x_train = imputed_train[inputs] if inputs else None
        self.x_val = imputed_val[inputs] if inputs else None

        print(f"Total data: {self.data.shape}\nTrain data: {self.x_train.shape}\nVal data: {self.x_val.shape}")

        return self.x_train, self.y_train, self.x_val, self.y_val

    def preprocessed_check(self):
        preprocessed_plot = self.util.plot_lines(self.y_train) + self.util.plot_lines(self.y_val)
        return preprocessed_plot

    def corr_check(self):
        corr = self.x_train.corr()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(corr, cmap='coolwarm')
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.index)
        # plt.show()

    def get_model(self, model_name, params):
        models = {'LR': self.linear_reg, 'Ridge': self.ridge, 'Lasso': self.lasso,
                  'DecisionTreeRegressor': self.dt,
                  'ARIMA': self.arima,
                  'DNN': self.dnn,
                  'LSTM': self.lstm}
        return models.get(model_name, self.linear_reg)(params)

    def lstm_model(self, params, input_shape):
        regressor = Sequential([
            LSTM(params['units'], return_sequences=True, input_shape=input_shape), Dropout(0.2),
            LSTM(params['units'], return_sequences=True), Dropout(0.2),
            LSTM(params['units'], return_sequences=True), Dropout(0.2),
            LSTM(params['units']), Dropout(0.2),
            Dense(1)
        ])
        regressor.compile(optimizer=params['optimizer'], loss='mean_squared_error')
        return regressor

    def linear_reg(self, params=None): return Model(LinearRegression(**params))
    def ridge(self, params=None): return Model(Ridge(**params))
    def lasso(self, params=None): return Model(Lasso(**params))
    def dt(self, params=None): return Model(DecisionTreeRegressor(**params))
    def rf(self, params=None): return Model(RandomForestRegressor(**params))
    def lg(self, params=None): return Model(LGBMRegressor(**params))
    def xg(self, params=None): return Model(XGBRegressor(**params))

    def arima(self, params=None):
        p = params.get('p', 1)
        d = params.get('d', 0)
        q = params.get('q', 0)
        trend = params.get('trend', 'c')
        return Model(ARIMA(order=(p, d, q), trend=trend, exog=self.x_train, endog=self.y_train))

    def dnn(self, params=None):
        regressor = Sequential([
            Dense(params['units'], input_shape=(self.x_train.shape[1],), activation='relu'),
            Dense(1, activation='linear')
        ])
        regressor.compile(optimizer=params['optimizer'], loss='mse')
        return Model(regressor)

    def lstm(self, params=None):
        regressor = Sequential([
            LSTM(params['units'], input_shape=(self.x_train.shape[1], 1), activation='tanh'),
            Dense(1, activation='linear')
        ])
        regressor.compile(optimizer=params['optimizer'], loss='mse')
        return Model(regressor)

    def get_predictions(self, model):
        if 'Results' in str(type(model)):
            raw_prediction = model.forecast(steps=self.y_val.shape[0], exog=self.x_val).values
            preds = pd.DataFrame(raw_prediction, columns=self.y_train.columns, index=self.y_val.index)
        else:
            raw_prediction = model.predict(self.x_val)
            preds = pd.DataFrame(raw_prediction, columns=self.y_train.columns, index=self.y_val.index)
        targets = self.y_train.columns
        t = self.preprocess.transform_types[targets[0]]
        if self.preprocess.transform_types and isinstance(t['Transform'], str) and t['Transform']!='':
            prediction = self.preprocess.revert_transform(data=preds, target=targets[0], original_data=self.data, transform=t['Transform'])
        else:
            prediction = pd.DataFrame(t.inverse_transform(raw_prediction.reshape(-1, 1)),
                                      columns=self.y_train.columns, index=self.y_val.index)
        return prediction

    def train_and_evaluate(self, model_object):
        metrics = {}
        model = model_object.model
        prediction = None
        if model_object.type == MLConfig.statsmodel:
            model_fit = model.fit()
            prediction = self.get_predictions(model_fit)
            self.model_objects.append(model_fit)
        elif model_object.type == MLConfig.sklearn:
            model.fit(self.x_train, self.y_train)
            self.model_objects.append(model)
        elif model_object.type == MLConfig.dl:
            model.fit(self.x_train, self.y_train, epochs=g_epochs, batch_size=g_batch_size)
            self.model_objects.append(model)

        if prediction is None:
            prediction = self.get_predictions(model)
        metric_list = MLConfig.ts_metrics

        for m in metric_list:
            target = self.data.columns[0]
            y_val_true = self.data[target].tail(self.y_val.shape[0])
            metrics[m] = round(self.metric_operator.calculate_metric(m, y_val_true, prediction), 5)
        return metrics

    def run_models(self, models):
        self.scores = {}
        for model in models:
            model_name = type(model).__name__
            metrics = self.train_and_evaluate(model)
            self.scores[model.name] = metrics

