import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score


def gridsearch_with_cv(train: pd.DataFrame, test: pd.DataFrame, test_y: pd.Series, train_y: pd.Series,
                       parameters_grids: dict, estimator, cv_folds, num_params_iter_max=10, metric='roc_auc'):
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(estimator,
                                       param_distributions=parameters_grids,
                                       n_iter=num_params_iter_max,
                                       scoring=metric,
                                       n_jobs=4,
                                       cv=strat_kfold.split(train, train_y), verbose=0, random_state=1001)

    random_search.fit(train, train_y, eval_set=[[test, test_y]])
    return random_search


def plot_variable_importance(model, feature_names: list):
    sorted_idx = model.feature_importances_.argsort()
    fig = plt.figure()

    plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
    plt.xlabel("Xgboost Feature Importance")
    plt.ylabel('ylabel', fontsize=8)
    plt.plot(fig)


def feature_selection(model_type, train_data: pd.DataFrame, train_y: pd.Series, test_data: pd.DataFrame,
                      test_y: pd.Series, metric=f1_score):
    model = model_type().fit(train_data, train_y)
    thresholds = np.sort(model.feature_importances_)
    results = []
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_x_train = selection.transform(train_data)
        # train model
        selection_model = model_type()
        selection_model.fit(select_x_train, train_y)
        # eval model
        select_x_test = selection.transform(test_data)
        predictions = selection_model.predict(select_x_test)
        metric_loop = metric(test_y, predictions)
        selected_features = select_x_train
        results.append({"treshold": thresh, "number of features": selected_features.shape[1], "metric": metric_loop,
                        "selected_features": ",".join(
                            train_data.columns[selection.get_support()])})
        print("Thresh=%.3f, n=%d, Metric: %.2f%%" % (thresh, selected_features.shape[1], metric_loop * 100.0))
    return results
