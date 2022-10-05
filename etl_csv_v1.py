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
        ('clf', Lasso(tol=1e-2, max_iter=5000))
    ])

    feature_selection = SelectFromModel(GridSearchCV(pipeline,
                                                     {'clf__alpha': np.arange(
                                                         0.1, 10, 0.1)},
                                                     cv=5, scoring="neg_mean_squared_error", verbose=0, n_jobs=6), importance_getter='best_estimator_.named_steps.clf.coef_')

    feature_selection.fit(X_train, y_train)

    params = feature_selection.get_support(
    )
    print(df_x.loc[:, [not i for i in params]].columns)

    # columns to remove
    return df_x.loc[:, [not i for i in params]].columns


def feature_combination():
    pass


def remove_atypical_values():
    pass


def invalid_correlated_features(df: pd.DataFrame):
    corr = df.corr()

    return [column for column in corr.columns if pd.isnull(corr[column])[1]]


def main(df: pd.DataFrame, options: list):
    data_options = {option: False for option in options}

    columns = df.columns

    if options["remove_time_features"]:
        print("Removing time features")
        df = remove_features(df, ['CaptureTime', 'SensorTime'])

        data_options["remove_time_features"] = True
    if options["generic_features"]:
        print("Selecting generic Features")
        # last position of generic features (areaFeatCount is not a generic feature)
        last_pos = df.columns.get_loc("areaFeatCount")
        df = remove_features(df, columns[last_pos:])

        data_options["generic_features"] = True
    if options["remove_invalid_correlated_features"]:
        print("Removing invalid correlated features")
        to_drop = invalid_correlated_features(df)
        df = remove_features(df, to_drop)

        data_options["remove_invalid_correlated_features"] = True
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
    df_options.to_csv("dataset_clean/options_csv_v1_etl.csv", index=False)

    return df


if __name__ == "__main__":
    df_PlatteRiver = pd.read_csv(
        "dataset/2012_2019_PlatteRiverWeir_features_merged_all.csv", index_col=0)
    df_PlatteRiver.columns = df_PlatteRiver.columns.str.replace(' ', '')

    # remove the non necessary features
    df_PlatteRiver = remove_features(
        df_PlatteRiver, ['Filename', 'Agency', 'SiteNumber', 'TimeZone', 'CalcTimestamp', 'width', 'height'])

    df_PlatteRiver = main(df_PlatteRiver, {'generic_features': False,
                                           'remove_atypical_values': False, 'feature_combination': False, 'remove_feature_selection': True, 'remove_time_features': True, 'remove_invalid_correlated_features': False})

    df_PlatteRiver.to_csv(
        "dataset_clean/PlatteRiverWeir_features_v1_clean.csv", index=False)
