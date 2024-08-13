import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import shap


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Plots the distribution of all numeric features in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be plotted.

    Returns:
        None
    """
    numeric_features = df.select_dtypes(include=["number"])
    num_features = len(numeric_features.columns)
    plots_per_row = 3
    num_rows = math.ceil(num_features / plots_per_row)
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_features.columns):
        sns.histplot(df[col], ax=axes[i], zorder=2, edgecolor="white", linewidth=0.5)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """
    Plots the distribution of all categorical features in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be plotted.

    Returns:
        None
    """
    categorical_features = df.select_dtypes(include=["category"])
    num_features = len(categorical_features.columns)
    plots_per_row = 3
    num_rows = math.ceil(num_features / plots_per_row)
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, col in enumerate(categorical_features.columns):
        sns.countplot(x=df[col], ax=axes[i], zorder=2)
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def bootstrap_median_diff(
    data1: np.ndarray, data2: np.ndarray, num_samples: int = 10000
) -> float:
    """
    Perform a bootstrap test to compare the difference in medians between two datasets.

    Args:
        data1 (np.ndarray): The first dataset.
        data2 (np.ndarray): The second dataset.
        num_samples (int): The number of bootstrap samples to generate (default is 10,000).

    Returns:
        float: The p-value of the bootstrap test.
    """
    observed_diff = np.median(data1) - np.median(data2)
    combined = np.concatenate([data1, data2])
    boot_diffs = []
    for _ in range(num_samples):
        boot_data1 = np.random.choice(combined, size=len(data1), replace=True)
        boot_data2 = np.random.choice(combined, size=len(data2), replace=True)
        boot_diff = np.median(boot_data1) - np.median(boot_data2)
        boot_diffs.append(boot_diff)
    p_value = np.sum(np.abs(boot_diffs) >= np.abs(observed_diff)) / num_samples
    return p_value


def numeric_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """
    Analyze the relationship between numeric predictors and a target feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    The function will:
        1. Generate histogram and boxplot visuals for each numeric predictor by target feature status.
        2. Perform a bootstrap hypothesis test for the difference in medians.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor in selected_columns:
        if predictor != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor} by {target_feature} Status",
                fontsize=13,
            )
            sns.histplot(
                data=temp_df,
                x=predictor,
                hue=target_feature,
                multiple="stack",
                edgecolor="white",
                ax=axs[0],
                zorder=2,
                linewidth=0.5,
                alpha=0.98,
            )
            axs[0].set_xlabel(predictor)
            axs[0].set_ylabel("Frequency")
            sns.boxplot(
                data=temp_df,
                y=predictor,
                x=target_feature,
                ax=axs[1],
                zorder=2,
            )
            axs[1].set_ylabel(predictor)
            axs[1].set_xlabel(target_feature)
            plt.tight_layout()
            plt.show()
            data_yes = temp_df[temp_df[target_feature] == "Yes"][predictor]
            data_no = temp_df[temp_df[target_feature] == "No"][predictor]
            p_value = bootstrap_median_diff(data_yes, data_no)
            print(f"{predictor} - {target_feature}:")
            print(f"Bootstrap hypothesis test p-value: {p_value:.2f}")


def test_distribution_similarity(df: pd.DataFrame, target_feature: str) -> None:
    """
    Tests the distribution similarity of numeric predictors against a target feature using the Kolmogorov-Smirnov test.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_columns.append(target_feature)
    for predictor in selected_columns:
        if predictor != target_feature:
            data_yes = df[df[target_feature] == "Yes"][predictor]
            data_no = df[df[target_feature] == "No"][predictor]
            ks_stat, ks_p_value = ks_2samp(data_yes, data_no)
            print(f"{predictor} - {target_feature}:")
            print(
                f"Kolmogorov-Smirnov test statistic: {ks_stat:.2f}, p-value: {ks_p_value:.2f}\n"
            )


def categorical_relationships(df: pd.DataFrame, target_feature: str) -> None:
    """
    Analyzes the relationship between categorical predictors and a target feature.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_feature (str): The name of the target feature column.

    The function will:
        1. Generate bar plots for each categorical predictor by target feature status.
        2. Perform Chi-square hypothesis test.

    Returns:
        None
    """
    selected_columns = df.select_dtypes(include=["category"]).columns.tolist()
    if target_feature not in selected_columns:
        selected_columns.append(target_feature)
    temp_df = df[selected_columns].copy()
    for predictor_name in temp_df.columns:
        if predictor_name != target_feature:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(
                f"Distribution of {predictor_name} by {target_feature} Status",
                fontsize=13,
            )
            crosstab = pd.crosstab(temp_df[predictor_name], temp_df[target_feature])
            crosstab.plot(
                kind="bar",
                stacked=True,
                ax=axs[0],
                edgecolor="white",
                linewidth=0.5,
            )
            axs[0].grid(axis="x")
            axs[0].set_xlabel(predictor_name)
            axs[0].set_ylabel("Count")
            axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=0)
            for patch in axs[0].patches:
                patch.set_zorder(2)
            crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0)
            crosstab_normalized.plot(
                kind="bar", stacked=True, ax=axs[1], edgecolor="white", linewidth=0.5
            )
            axs[1].grid(axis="x")
            axs[1].set_ylabel("Proportion")
            axs[1].set_xlabel(predictor_name)
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)
            axs[1].legend().set_visible(False)
            for patch in axs[1].patches:
                patch.set_zorder(2)
            plt.tight_layout()
            plt.show()
            _, p_value, _, _ = chi2_contingency(crosstab)
            print(
                f"{predictor_name} - {target_feature}:\nChi-Square test p-value: {p_value:.2f}\n"
            )


def phik_heatmap(df: pd.DataFrame, size: float = 1) -> None:
    """
    Generate a heatmap of the Phik correlation matrix for the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        size (float, optional): Scaling factor for the figure size. Default is 1.

    Returns:
        None
    """
    colors = ["#4C72B0", "#DD8452"]
    n_bins = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_blue", list(zip(n_bins, colors)))
    numeric_features = df.select_dtypes(include=["float64"])
    phik_matrix = df.phik_matrix(interval_cols=numeric_features)
    mask = np.triu(np.ones_like(phik_matrix, dtype=bool))
    plt.figure(figsize=(7 * size, 5 * size))
    sns.heatmap(
        phik_matrix,
        mask=mask,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar=False,
    )
    plt.grid(False)
    plt.title(r"$\phi_K$ Correlation Heatmap Between All Variables")


def vif(df: pd.DataFrame, target_variable: str) -> None:
    """
    Calculate and display the Variance Inflation Factor (VIF) for predictor variables in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_variable (str): The name of the target variable to be excluded from the VIF calculation.

    Returns:
        None
    """
    X = df.drop(columns=target_variable)
    y = df[target_variable]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="category").columns.tolist()
    preprocessor_vif = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    X_preprocessed = preprocessor_vif.fit_transform(X)
    X_preprocessed_const = add_constant(X_preprocessed)
    numeric_feature_names = numeric_features
    categorical_feature_names = list(
        preprocessor_vif.named_transformers_["cat"].get_feature_names_out(
            categorical_features
        )
    )
    all_feature_names = ["const"] + numeric_feature_names + categorical_feature_names
    vif_data = pd.DataFrame()
    vif_data["Feature"] = all_feature_names
    vif_data["VIF"] = [
        variance_inflation_factor(X_preprocessed_const, i)
        for i in range(X_preprocessed_const.shape[1])
    ]
    display(vif_data)


def plot_confusion(
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plots the confusion matrix for test dataset.

    Args:
        y_test (pd.Series): Actual test data.
        y_test_pred (np.ndarray): Predicted values for testing set.

    Returns:
        None
    """
    colors = ["#FFFFFF", "#DD8452"]
    n_bins = [0, 1]
    cmap = LinearSegmentedColormap.from_list("custom_blue", list(zip(n_bins, colors)))
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(3, 3))
    sns.heatmap(
        conf_matrix_test,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        linewidths=0.7,
    )
    plt.grid(False)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def algorithm_selection_pca_weighted(
    X: pd.DataFrame, y_encoded: np.ndarray, models: dict
) -> None:
    """
    Perform initial algorithm selection for binary classification using PCA and sample weights.

    Args:
        X (pd.DataFrame): Feature matrix.
        y_encoded (np.ndarray): Encoded target variable.
        models (dict): Dictionary of models to be evaluated.

    Returns:
        None
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, pos_label=1),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "pr_auc": make_scorer(average_precision_score, pos_label=1),
        "precision": make_scorer(precision_score, pos_label=1),
        "recall": make_scorer(recall_score, pos_label=1),
    }
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="category").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    preprocessor_drop_category = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_encoded)
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    for model_name, single_model in models.items():
        if model_name in ["LogisticRegression", "SVM"]:
            pcr_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_drop_category),
                    ("pca", PCA(n_components="mle")),
                    ("classifier", single_model),
                ]
            )
        else:
            pcr_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("pca", PCA(n_components="mle")),
                    ("classifier", single_model),
                ]
            )
        if model_name == "GradientBoosting":
            cv_scores = cross_validate(
                pcr_pipeline,
                X,
                y_encoded,
                cv=cv,
                scoring=scoring,
                fit_params={"classifier__sample_weight": sample_weights},
                return_train_score=True,
            )
        else:
            cv_scores = cross_validate(
                pcr_pipeline,
                X,
                y_encoded,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
            )
        result = {
            "model": model_name,
            "pr_auc": cv_scores["test_pr_auc"].mean(),
            "roc_auc": cv_scores["test_roc_auc"].mean(),
            "f1": cv_scores["test_f1"].mean(),
            "precision": cv_scores["test_precision"].mean(),
            "recall": cv_scores["test_recall"].mean(),
            "validation_accuracy": cv_scores["test_accuracy"].mean(),
            "train_accuracy": cv_scores["train_accuracy"].mean(),
        }
        results.append(result)
    results_df = pd.DataFrame(results).set_index("model")
    results_df.index.name = None
    display(results_df)


def algorithm_selection_pca_smote(
    X: pd.DataFrame, y_encoded: np.ndarray, models: dict
) -> None:
    """
    Perform initial algorithm selection for binary classification using PCA and SMOTE oversampling.

    Args:
        X (pd.DataFrame): Feature matrix.
        y_encoded (np.ndarray): Encoded target variable.
        models (dict): Dictionary of models to be evaluated.

    Returns:
        None
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, pos_label=1),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "pr_auc": make_scorer(average_precision_score, pos_label=1),
        "precision": make_scorer(precision_score, pos_label=1),
        "recall": make_scorer(recall_score, pos_label=1),
    }
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="category").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )
    preprocessor_drop_category = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    for model_name, single_model in models.items():
        if model_name in ["LogisticRegression", "SVM"]:
            pcr_pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor_drop_category),
                    ("smote", SMOTE(random_state=5)),
                    ("pca", PCA(n_components="mle")),
                    ("classifier", single_model),
                ]
            )
        else:
            pcr_pipeline = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=5)),
                    ("pca", PCA(n_components="mle")),
                    ("classifier", single_model),
                ]
            )
        cv_scores = cross_validate(
            pcr_pipeline, X, y_encoded, cv=cv, scoring=scoring, return_train_score=True
        )
        result = {
            "model": model_name,
            "pr_auc": cv_scores["test_pr_auc"].mean(),
            "roc_auc": cv_scores["test_roc_auc"].mean(),
            "f1": cv_scores["test_f1"].mean(),
            "precision": cv_scores["test_precision"].mean(),
            "recall": cv_scores["test_recall"].mean(),
            "validation_accuracy": cv_scores["test_accuracy"].mean(),
            "train_accuracy": cv_scores["train_accuracy"].mean(),
        }
        results.append(result)
    results_df = pd.DataFrame(results).set_index("model")
    results_df.index.name = None
    display(results_df)


def optimize_logistic_regression(
    X: pd.DataFrame, y_encoded: np.ndarray, param_grid: list[dict[str, any]]
) -> Pipeline:
    """
    Tune logistic regression model using GridSearchCV, perform cross-validation, and visualize the results.

    Args:
        X (pd.DataFrame): Feature dataframe with numeric and categorical columns.
        y_encoded (np.ndarray): Encoded target variable array.
        param_grid (list[dict[str, any]]): List of dictionaries specifying hyperparameter grid for GridSearchCV.

    Returns:
        Pipeline: The best pipeline from GridSearchCV.
    """
    scoring = {
        "accuracy": "accuracy",
        "f1": make_scorer(f1_score, pos_label=1),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "pr_auc": make_scorer(average_precision_score, pos_label=1),
        "precision": make_scorer(precision_score, pos_label=1),
        "recall": make_scorer(recall_score, pos_label=1),
    }
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include="category").columns.tolist()
    preprocessor_drop_category = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("log", FunctionTransformer(np.log1p, validate=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )
    pipeline_logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor_drop_category),
            ("classifier", LogisticRegression()),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    grid_search = GridSearchCV(
        estimator=pipeline_logistic_regression,
        param_grid=param_grid,
        cv=cv,
        scoring="average_precision",
    )
    grid_search.fit(X, y_encoded)
    best_params = grid_search.best_params_
    print(f"Best hyperparameters: {best_params}\n")

    best_pipeline = grid_search.best_estimator_
    cv_scores = cross_validate(
        best_pipeline, X, y_encoded, cv=cv, scoring=scoring, return_train_score=True
    )
    result = {
        "model": "Logistic Regression",
        "pr_auc": cv_scores["test_pr_auc"].mean(),
        "roc_auc": cv_scores["test_roc_auc"].mean(),
        "f1": cv_scores["test_f1"].mean(),
        "precision": cv_scores["test_precision"].mean(),
        "recall": cv_scores["test_recall"].mean(),
        "validation_accuracy": cv_scores["test_accuracy"].mean(),
        "train_accuracy": cv_scores["train_accuracy"].mean(),
    }
    results_df = pd.DataFrame([result]).set_index("model")
    results_df.index.name = None
    display(results_df)

    y_pred = cross_val_predict(best_pipeline, X, y_encoded, cv=cv)
    plot_confusion(y_encoded, y_pred)

    coefficients = best_pipeline.named_steps["classifier"].coef_[0]
    preprocessor_returned = best_pipeline.named_steps["preprocessor"]
    numeric_features = preprocessor_returned.transformers_[0][2]
    categorical_features = preprocessor_returned.transformers_[1][
        1
    ].get_feature_names_out(preprocessor_returned.transformers_[1][2])
    feature_names = np.concatenate([numeric_features, categorical_features])
    sorted_indices = np.argsort(coefficients)[::-1]
    sorted_coefficients = coefficients[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]
    plt.figure(figsize=(10, 4))
    plt.barh(np.arange(len(sorted_coefficients)), sorted_coefficients, zorder=2)
    plt.yticks(np.arange(len(sorted_coefficients)), sorted_feature_names)
    plt.xlabel("Coefficient Value (Logarithmic Axis)")
    plt.ylabel("Features")
    plt.title("Logistic Regression Coefficients")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    classifier = best_pipeline.named_steps["classifier"]
    X_transformed = preprocessor_returned.transform(X)
    explainer = shap.Explainer(classifier, X_transformed)
    shap_values = explainer(X_transformed)
    shap.summary_plot(shap_values, features=X_transformed, feature_names=feature_names)
    return best_pipeline


def plot_final_evaluation_curves(y_test: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """
    Plot ROC and Precision-Recall curves, and threshold performance curves for evaluation.

    Args:
        y_test (np.ndarray): Binary true labels.
        y_pred_prob (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        None
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(
        fpr,
        tpr,
        lw=2,
        color="#DD8452",
        label=f"Positive class ROC curve (area = {roc_auc_score(y_test, y_pred_prob):.2f})",
        zorder=2,
    )
    y_pred_prob_neg = 1 - y_pred_prob
    fpr_neg, tpr_neg, _ = roc_curve(1 - y_test, y_pred_prob_neg)
    plt.plot(
        fpr_neg,
        tpr_neg,
        lw=2,
        color="#4C72B0",
        label=f"Negative class ROC curve (area = {roc_auc_score(1 - y_test, y_pred_prob_neg):.2f})",
        zorder=2,
    )
    plt.plot([0, 1], [0, 1], color="Grey", lw=2, linestyle="--", zorder=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.subplot(1, 2, 2)
    plt.plot(
        recall,
        precision,
        lw=2,
        color="#DD8452",
        label=f"Positive class PR curve (area = {average_precision_score(y_test, y_pred_prob):.2f})",
        zorder=2,
    )
    precision_neg, recall_neg, _ = precision_recall_curve(1 - y_test, y_pred_prob_neg)
    plt.plot(
        recall_neg,
        precision_neg,
        lw=2,
        color="#4C72B0",
        label=f"Negative class PR curve (area = {average_precision_score(1 - y_test, y_pred_prob_neg):.2f})",
        zorder=2,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    # plt.savefig('ROC_PR_curves.png')
    plt.show()

    thresholds = np.arange(0.0, 1.0, 0.01)
    precisions = []
    recalls = []
    f1s = []
    for threshold in thresholds:
        y_pred_threshold = (y_pred_prob >= threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred_threshold))
        recalls.append(recall_score(y_test, y_pred_threshold))
        f1s.append(f1_score(y_test, y_pred_threshold))
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, recalls, label="recall", color="#1f77b4")
    plt.plot(thresholds, precisions, label="precision", color="#2ca02c")
    plt.plot(thresholds, f1s, label="f1", color="#ff7f0e")
    plt.xlabel("Cutoff")
    plt.ylabel("Score")
    plt.title("Threshold Performance Curve")
    plt.legend(loc="best")
    plt.grid(True)
    # plt.savefig('threshold_performance_curves.png')
    plt.show()
