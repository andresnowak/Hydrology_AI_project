import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


def remove_features(df: pd.DataFrame, features: list):
    return df.drop(columns=features)


def feature_selection(df_x: pd.DataFrame, df_y: pd.DataFrame):
    X_train, _, y_train, _ = train_test_split(
        df_x, df_y, test_size=0.33, random_state=0)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(tol=1e-2, max_iter=10000))
    ])

    feature_selection = GridSearchCV(pipeline,
                                     {'model__alpha': np.arange(
                                         0.1, 10, 0.1)},
                                     cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=8)

    feature_selection.fit(X_train, y_train)

    coefficients = feature_selection.best_estimator_.named_steps['model'].coef_[
        0]
    importance = np.abs(coefficients)

    return df_x.loc[:, importance == 0].columns  # columns to remove


def feature_combination():
    pass


def remove_atypical_values():
    pass


def main(df: pd.DataFrame, options: list):
    data_options = {option: False for option in options}

    columns = df.columns

    if options["remove_time_features"]:
        df = remove_features(df, ['CaptureTime', 'SensorTime'])

        data_options["remove_time_features"] = True
    if options["generic_features"]:
        print("Selecting generic Features")
        # last position of generic features (areaFeatCount is not a generic feature)
        last_pos = df.columns.get_loc("areaFeatCount")
        df = remove_features(df, columns[last_pos:])

        data_options["generic_features"] = True
    if options["feature_combination"]:
        pass
    if options["remove_atypical_values"]:
        pass
    if options["remove_feature_selection"]:
        print('Removing feature selection')

        df_y = df[["Stage", "Discharge"]]
        df_x = df.drop(
            columns=["Stage", "Discharge"])

        columns_remove = feature_selection(df_x, df_y)

        df = remove_features(df, columns_remove)

        data_options["remove_feature_selection"] = "Lasso"

    print(data_options)
    df_options = pd.DataFrame(data_options, index=[0])
    df_options.to_csv("dataset_clean/options_csv_v1_etl.csv")

    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "dataset/2012_2019_PlatteRiverWeir_features_merged_all.csv", index_col=0)
    df.columns = df.columns.str.replace(' ', '')

    # remove the non necessary features
    df = remove_features(
        df, ['Filename', 'Agency', 'SiteNumber', 'TimeZone', 'CalcTimestamp', 'width', 'height'])

    df = main(df, {'generic_features': True,
                   'remove_atypical_values': False, 'feature_combination': False, 'remove_feature_selection': True, 'remove_time_features': True})

    df.to_csv("dataset_clean/PlatteRiverWeir_features_v1_clean.csv")
