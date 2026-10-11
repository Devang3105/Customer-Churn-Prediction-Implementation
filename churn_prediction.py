from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FILE = Path("telecom_customer_churn.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42


def load_data():
    """Load the source CSV and report useful initial information."""
    try:
        df = pd.read_csv(DATA_FILE)
        print("Dataset loaded successfully.")
        print("Dataset shape:", df.shape)
        print("\nFirst five rows:")
        print(df.head())
        print("\nData types and non-null counts:")
        df.info()
        print("\nMissing values (top 15):")
        print(df.isna().sum().sort_values(ascending=False).head(15))
        return df
    except FileNotFoundError:
        print(f"ERROR: Cannot find {DATA_FILE}.")
        print("Place telecom_customer_churn.csv in the same folder as this script.")
        return None
    except Exception as exc:
        print("Could not load the dataset:", exc)
        return None


def prepare_target(df):
    """Convert the dataset's churn/status column into a binary target."""
    df = df.copy()

    if "Customer Status" in df.columns:
        status = df["Customer Status"].astype(str).str.strip().str.lower()
        # The PPT describes the statuses Stayed, Churned, and Joined.
        # Churned is positive; Stayed/Joined are treated as not churned.
        y = status.eq("churned").astype(int)
        valid = status.isin(["churned", "stayed", "joined"])
        if valid.sum() == 0:
            raise ValueError("Customer Status does not contain Churned/Stayed/Joined values.")
        df = df.loc[valid].copy()
        y = y.loc[valid]
        target_name = "Customer Status"
    elif "Churn" in df.columns:
        raw = df["Churn"].astype(str).str.strip().str.lower()
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        y = raw.map(mapping)
        valid = y.notna()
        df = df.loc[valid].copy()
        y = y.loc[valid].astype(int)
        target_name = "Churn"
    else:
        raise ValueError(
            "Target column not found. Expected 'Customer Status' or 'Churn'."
        )

    if y.nunique() != 2:
        raise ValueError("Binary churn modeling requires both churn classes in the dataset.")

    return df, y.astype(int), target_name


def engineer_features(df):
    """Create the feature-engineering columns described in the project PPT."""
    df = df.copy()

    if "Tenure in Months" in df.columns:
        tenure = pd.to_numeric(df["Tenure in Months"], errors="coerce")
        df["TenureGroup"] = pd.cut(
            tenure,
            bins=[0, 12, 36, 72],
            labels=["Short", "Medium", "Long"],
            include_lowest=True
        )

    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
    ]
    existing_services = [col for col in service_cols if col in df.columns]
    if existing_services:
        df["ServiceCount"] = (
            df[existing_services]
            .apply(lambda col: col.astype(str).str.strip().str.lower().eq("yes"))
            .sum(axis=1)
        )

    if "Tenure in Months" in df.columns and "Total Revenue" in df.columns:
        tenure = pd.to_numeric(df["Tenure in Months"], errors="coerce")
        revenue = pd.to_numeric(df["Total Revenue"], errors="coerce")
        df["RevenuePerMonth"] = np.where(
            tenure > 0, revenue / tenure, revenue
        )

    # Convert common numeric-looking fields safely.
    for col in ["Total Charges", "Monthly Charge", "Total Revenue",
                "Tenure in Months", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Identifiers generally do not help generalize to unseen customers.
    id_cols = [
        col for col in df.columns
        if col.strip().lower() in {"customerid", "customer id", "id"}
    ]
    df = df.drop(columns=id_cols, errors="ignore")
    return df


def save_eda(df, y):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    counts = y.map({0: "Not Churned", 1: "Churned"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, color="steelblue")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer status")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=200)
    plt.close()

    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
        plt.close()

    # Statistical examples from the project: t-test and chi-square.
    if "Monthly Charge" in df.columns:
        charges = pd.to_numeric(df["Monthly Charge"], errors="coerce")
        stayed = charges[(y == 0) & charges.notna()]
        churned = charges[(y == 1) & charges.notna()]
        if len(stayed) >= 2 and len(churned) >= 2:
            result = stats.ttest_ind(churned, stayed, equal_var=False)
            print("\nWelch t-test: Monthly Charge (Churned vs Not Churned)")
            print(f"t-statistic = {result.statistic:.4f}, p-value = {result.pvalue:.6g}")

    if "Contract" in df.columns:
        table = pd.crosstab(df["Contract"], y)
        if table.shape[0] >= 2 and table.shape[1] == 2:
            chi2, p_value, dof, _ = stats.chi2_contingency(table)
            print("\nChi-square test: Contract vs Churn")
            print(f"chi-square = {chi2:.4f}, p-value = {p_value:.6g}, df = {dof}")


def main():
    df = load_data()
    if df is None:
        return

    try:
        df, y, target_name = prepare_target(df)
    except ValueError as exc:
        print("Target preparation error:", exc)
        return

    print("\nTarget column:", target_name)
    print("Target distribution (0 = Not Churned, 1 = Churned):")
    print(y.value_counts().sort_index())

    # Do not use the original target as a model input.
    feature_df = df.drop(
        columns=[target_name, "Churn", "Churn Category", "Churn Reason"],
        errors="ignore"
    )
    feature_df = engineer_features(feature_df)

    # Remove columns that are entirely empty.
    feature_df = feature_df.dropna(axis=1, how="all")
    save_eda(feature_df, y)

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        col for col in feature_df.columns if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.20, random_state=RANDOM_STATE,
        stratify=stratify
    )

    results = []
    fitted_models = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
        }
        if hasattr(pipeline, "predict_proba") and y_test.nunique() == 2:
            try:
                row["ROC AUC"] = roc_auc_score(
                    y_test, pipeline.predict_proba(X_test)[:, 1]
                )
            except ValueError:
                row["ROC AUC"] = np.nan
        results.append(row)
        fitted_models[name] = pipeline
        print(f"\n{name}")
        print(classification_report(
            y_test, predictions,
            labels=[0, 1],
            target_names=["Not Churned", "Churned"],
            zero_division=0
        ))

    metrics = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print("\nModel comparison:")
    print(metrics.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plot_data = metrics.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric", value_name="Score"
    )
    sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric", palette="deep")
    plt.ylim(0, 1)
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close()

    best_name = metrics.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)
    print(f"\nBest model by F1 Score: {best_name}")
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Churned", "Churned"],
        yticklabels=["Not Churned", "Churned"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Export test-set predictions, including IDs when present in the source.
    prediction_output = pd.DataFrame({
        "Actual_Churn": y_test.map({0: "Not Churned", 1: "Churned"}),
        "Predicted_Churn": pd.Series(
            best_predictions, index=y_test.index
        ).map({0: "Not Churned", 1: "Churned"})
    })
    prediction_output.to_csv(OUTPUT_DIR / "customer_predictions.csv", index_label="Row_Index")

    # Random Forest feature importance is useful for explaining model behavior.
    rf_pipeline = fitted_models["Random Forest"]
    rf_model = rf_pipeline.named_steps["model"]
    transformed_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.Series(
        rf_model.feature_importances_, index=transformed_names
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    print(f"\nOutputs saved in: {OUTPUT_DIR.resolve()}")
    print("Completed. Remember: evaluation metrics are from this dataset split,")
    print("and feature importance describes model behavior, not causation.")


if __name__ == "__main__":
    main()
