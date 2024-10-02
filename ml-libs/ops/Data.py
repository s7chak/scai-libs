from datetime import datetime, timezone

import numpy as np
import ops.MLConfig as MLConfig
import polars as pl
from ops.Utils import Util, time_this
from sklearn.preprocessing import StandardScaler


class DataPrepper:
    def __init__(self, source_filename, input_columns, output_columns, transform_dict=None, split_ratio=None, split_date=None):
        self.source_filename = source_filename
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.transform_dict = transform_dict
        self.split_ratio = split_ratio
        self.split_date = split_date
        self.metadata = {'file': self.source_filename}
        self.modeling_type = MLConfig.reg_modeltype
        self.raw_df = self.load_file()
        self.validate_data()

    @time_this
    def load_file(self):
        if self.source_filename.endswith('.csv'):
            return pl.read_csv(self.source_filename)
        elif self.source_filename.endswith('.xlsx'):
            return pl.read_excel(self.source_filename)
        elif self.source_filename.endswith('.parquet'):
            return pl.read_parquet(self.source_filename)
        else:
            raise ValueError(f"Unsupported file format: {self.source_filename}")

    @time_this
    def validate_data(self):
        self.x_train, self.x_val, self.y_train, self.y_val = [None]*4
        include_columns = self.input_columns + self.output_columns
        # Identify the date column
        date_column = next((col for col in self.raw_df.columns if col.lower() in MLConfig.date_col_names), None)
        if date_column:
            self.raw_df = self.raw_df.with_columns(pl.col(date_column).alias(MLConfig.date_column))
            include_columns.append(MLConfig.date_column)
            self.modeling_type = MLConfig.tsa_modeltype
            self.raw_df = self.raw_df.with_columns(
                pl.col(date_column).str.to_datetime().dt.convert_time_zone("UTC").alias(MLConfig.date_column)
            ).sort(MLConfig.date_column)
        else:
            self.modeling_type = MLConfig.reg_modeltype
        self.raw_df = self.raw_df.select(include_columns)
        # Variable Metadata
        total_rows = len(self.raw_df)
        self.var_metadata = {
            col: {
                'dtype': str(self.raw_df[col].dtype),
                'ctype': 'categorical' if self.raw_df[col].n_unique() / total_rows < 0.1 else 'continuous',
                'name': col,
                'count': self.raw_df[col].drop_nulls().count()
            }
            for col in self.raw_df.columns
        }
        self.metadata['shape'] = self.raw_df.shape
        self.metadata['var_metadata'] = self.var_metadata

    @time_this
    def run_preprocessing(self, transform_dict=None, split_ratio=None, split_date=None):
        if transform_dict:
            self.transform_dict = transform_dict
        if split_ratio:
            self.split_ratio = split_ratio
        if split_date:
            self.split_date = split_date

        if not self.transform_dict:
            raise ValueError("No transform information provided, please supply transform_dict.")
        if not (self.split_date or self.split_ratio):
            print('No split information provided.')


        preprocessor = Preprocess()
        preprocessor.data_preprocessing(
            data=self,
            split=self.split_date or self.split_ratio,
            target=self.output_columns,
            inputs=self.input_columns,
            transform_dict=self.transform_dict
        )



    def show_metadata(self):
        for key, value in self.metadata.items():
            if key == 'var_metadata' and isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f" {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")

class Preprocess:
    def __init__(self):
        self.transform_types = {}
        self.util = Util()

    def data_preprocessing(self, data: DataPrepper, split: str = None, target: str = None, inputs: list = None, transform_dict: dict = None, dataset: pl.DataFrame = None):
        date_col = MLConfig.date_column
        raw_data = data.raw_df
        transformed_df = raw_data
        if split:
            split_data = self.util.check_split(split)
            if type(split_data)!=float:
                split_date = pl.lit(split_data)
                train = transformed_df.filter(pl.col(date_col) < split_date)
                val = transformed_df.filter(pl.col(date_col) >= split_date)
            else:
                train_size = int(len(transformed_df) * split_data)
                train = transformed_df.head(train_size)  # Select the first 'train_size' rows
                val = transformed_df.tail(len(transformed_df) - train_size)
        else:
            train = val = transformed_df

        transformed_train = self.transform(train, transform_dict)
        imputed_train = self.impute(transformed_train, transform_dict)
        data.x_train = imputed_train.select(inputs) if inputs else None
        data.y_train = imputed_train.select([target])
        print(f"Total data: {transformed_df.shape}")
        if split:
            transformed_val = self.transform(val, transform_dict)
            imputed_val = self.impute(transformed_val, transform_dict)
            data.x_val = imputed_val.select(inputs) if inputs else None
            data.y_val = imputed_val.select([target])
            print(f"Train data: {data.x_train.shape if data.x_train is not None else 'N/A'}\nVal data: {data.x_val.shape if data.x_val is not None else 'N/A'}")
        data.transformed_df = transformed_df

    def transform_column(self, transform, column: pl.Series):
        if isinstance(transform, StandardScaler):
            scaler = transform.fit(column.to_numpy().reshape(-1, 1))
            return pl.Series(scaler.transform(column.to_numpy().reshape(-1, 1)).flatten())
        elif 'diff' in transform:
            diff_order = int(transform[-1])
            return column.diff(diff_order).fill_null(0)
        elif transform == 'log_diff1':
            return np.log(column).diff().fill_null(0)
        elif transform == 'log':
            return np.log(column)
        else:
            raise ValueError(f"Unsupported transformation: {transform}")

    def transform(self, data: pl.DataFrame, transform_dict):
        transformed_data = data.select([])
        self.transform_types = transform_dict
        for column, settings in transform_dict.items():
            transform_method = settings.get('Transform')
            transformed_column = self.transform_column(transform_method, data[column]) if transform_method else data[column]
            transformed_data = transformed_data.with_columns(transformed_column.alias(column))

        return transformed_data

    def impute_column(self, impute_method, column):
        if impute_method == 'Mean':
            return column.fillna(column.mean())
        elif impute_method == 'Median':
            return column.fillna(column.median())
        elif impute_method == 'Linear':
            return column.interpolate(method='linear')
        elif impute_method == 'Spline':
            return column.interpolate(method='spline', order=2)
        else:
            raise ValueError(f"Unsupported imputation method: {impute_method}")

    def impute(self, data: pl.DataFrame, transform_dict):
        imputed_data = data.select([])  # Empty DataFrame to store imputed columns
        for column, settings in transform_dict.items():
            impute_method = settings.get('Impute')
            imputed_column = self.impute_column(impute_method, data[column]) if impute_method else data[column]
            imputed_data = imputed_data.with_columns(imputed_column.alias(column))
        return imputed_data

    def revert_transform(self, data: pl.DataFrame, target: str, transform=None, original_data: pl.DataFrame = None):
        if isinstance(transform, StandardScaler):
            return pl.DataFrame(transform.inverse_transform(data[target].to_numpy().reshape(-1, 1)).flatten(), schema=[target])

        reverted_y = data[target]

        if 'diff' in transform:
            diff_order = int(transform[-1])
            if original_data is not None:
                reverted_y = pl.concat([pl.Series([original_data[target][0]]), reverted_y])
            for _ in range(diff_order):
                reverted_y = reverted_y.cumsum()
            reverted_y = reverted_y[1:]

        elif transform == 'log':
            reverted_y = reverted_y.apply(np.exp)

        elif transform == 'log_diff':
            if original_data is not None:
                reverted_y = pl.concat([pl.Series([np.log(original_data[target][0])]), reverted_y])
            reverted_y = reverted_y.cumsum().apply(np.exp)[1:]

        return pl.DataFrame(reverted_y, schema=[target])