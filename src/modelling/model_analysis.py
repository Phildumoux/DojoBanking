import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import matplotlib.pyplot as plt
import shap


def plot_shapley_values(model, input_data: pd.DataFrame, sample_frac: float):
    sample_test = input_data.sample(frac=sample_frac)
    explainer = shap.Explainer(model.predict, sample_test)
    shap_values = explainer(sample_test)
    shap.summary_plot(shap_values, sample_test, plot_type="bar")
    shap.plots.beeswarm(shap_values)


def plot_feature_importance(model, features: list, output_path: str = "output/"):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(12, 12))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.title('Feature Importance')
    fig.savefig('{}_var_importance.png'.format(output_path))


def save_and_plot_partial_dependence_plots(model, train_data: pd.DataFrame, features: list,
                                           output_path: str = "output/pdp/"):
    for feature in features:
        pdp_plot = PartialDependenceDisplay.from_estimator(model, train_data, [feature])
        pdp_plot.figure_.savefig('{}{}_pdp.png'.format(output_path, feature))
