import pandas as pd

from ops.Data import DataPrepper
from ops.Models import TimeSeriesModeler
from ops.Utils import time_this
import ops.MLConfig as MLConfig

def run_modeling_test():
    data = pd.read_csv('../inputs/tstraining.csv')
    target = '^GSPC'
    inputs = ['AAPL', 'MSFT']
    split_year = '2019'

    modeler = TimeSeriesModeler()

    transform_dict = {
        'AAPL': {'Transform': 'diff1', 'Impute': 'Linear'},
        'MSFT': {'Transform': 'diff1', 'Impute': 'Linear'},
        '^GSPC': {'Transform': 'log_diff1', 'Impute': 'Linear'}
    }

    modeler.data_preprocessing(data, split_year, target, inputs, transform_dict)
    pp_plot = modeler.preprocessed_check()
    modeler.corr_check()
    models_to_run = [modeler.get_model('LR', {}),
                     modeler.get_model('ARIMA', {'p': 1}),
                     modeler.get_model('DNN', {'units': 30, 'optimizer': 'adam'}),
                     modeler.get_model('LSTM', {'units': 30, 'optimizer': 'adam'}),
                     ]

    modeler.run_models(models_to_run)
    print(modeler.scores)


ts_test_data = {'filename': '../inputs/tstraining.csv',
    'targets' : ['^GSPC'],
    'inputs' : ['AAPL', 'MSFT'],
    'split_date' : '2021-01-01',
    'transform_dict' : {
        'AAPL': {'Transform': 'diff1', 'Impute': 'Linear'},
        'MSFT': {'Transform': 'diff1', 'Impute': 'Linear'},
        '^GSPC': {'Transform': 'log_diff1', 'Impute': 'Linear'}
    }
}

reg_test_data = {'filename': '../inputs/regtraining.csv',
    'targets' : ['wind_speed'],
    'inputs' : ['meantemp', 'humidity'],
    'split_ratio' : '0.7',
    'transform_dict' : {
        'wind_speed': {'Transform': 'diff1', 'Impute': 'Linear'},
        'meantemp': {'Transform': 'diff1', 'Impute': 'Linear'},
        'humidity': {'Transform': 'diff1', 'Impute': 'Linear'}
    }
}

test_info = {MLConfig.tsa_modeltype: ts_test_data, MLConfig.reg_modeltype: reg_test_data}

@time_this
def run_new_test():

    test_run_type = MLConfig.reg_modeltype
    loaded_test_data = test_info[test_run_type]
    filename = loaded_test_data['filename']

    targets = loaded_test_data['targets']
    inputs = loaded_test_data['inputs']
    split_date = loaded_test_data['split_date'] if 'split_date' in loaded_test_data else None
    split_ratio = loaded_test_data['split_ratio'] if 'split_ratio' in loaded_test_data else None
    transform_dict = loaded_test_data['transform_dict']
    data = DataPrepper(source_filename=filename, input_columns=inputs, output_columns=targets)
    data.show_metadata()
    data.run_preprocessing(transform_dict=transform_dict, split_ratio=split_ratio, split_date=split_date)
    # eda = Explorer(data)
    # eda.generate_eda_report('eda-report.docx')



if __name__ == "__main__":
    run_new_test()