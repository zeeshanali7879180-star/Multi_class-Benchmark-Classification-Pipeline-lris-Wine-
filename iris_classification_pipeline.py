"""
PROJECT: Multiclass Benchmarks Classification Pipeline
DATASET: Iris
MODELS: Logistic Regression and Decision Tree
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/iris.csv"
)

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

TARGET = "species"

RANDOM_STATE = 42
TEST_SIZE = 0.20

sns.set_theme(style="whitegrid")


# ==========================================
# 2. LOAD DATASET
# ==========================================

def load_dataset():
    print("Loading Iris Dataset...")

    data = pd.read_csv(DATA_URL)

    print("\nDataset Loaded Successfully!")

    return data


# ==========================================
# 3. EXPLORATORY DATA ANALYSIS
# ==========================================

def perform_eda(data):
    print("\n========== DATASET INFORMATION ==========")

    print("\nFirst 5 Rows:")
    print(data.head())

    print("\nDataset Shape:")
    print(data.shape)

    print("\nDataset Information:")
    data.info()

    print("\nStatistical Summary:")
    print(data.describe())

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nSpecies Distribution:")
    print(data[TARGET].value_counts())

    # Pairplot
    pair_plot = sns.pairplot(
        data,
        vars=FEATURES,
        hue=TARGET,
        diag_kind="hist",
        corner=True
    )

    pair_plot.fig.suptitle(
        "Iris Dataset - Feature Pairplot",
        y=1.02
    )

    pair_plot.savefig(
        OUTPUT_DIR / "01_pairplot.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

def prepare_data(data):
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n========== DATA SPLITTING ==========")

    print("Training Samples:", len(X_train))
    print("Testing Samples:", len(X_test))

    return X_train, X_test, y_train, y_test


# ==========================================
# 5. BUILD CLASSIFICATION MODELS
# ==========================================

def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=4
        )
    }

    return models


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

def train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
):
    results = []

    class_names = sorted(y_train.unique())

    for model_name, model in models.items():
        print("\n===================================")
        print("MODEL:", model_name)
        print("===================================")

        # Model Training
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Evaluation Metrics
        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # Print Results
        print("\nAccuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions,
                labels=class_names,
                zero_division=0
            )
        )

        # Save Metrics
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Confusion Matrix
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=class_names
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        display.plot(
            cmap="Blues",
            values_format="d"
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(
            OUTPUT_DIR / f"confusion_matrix_{filename}.png",
            dpi=300
        )

        plt.show()

    # Benchmark Comparison
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    print("\n========== MODEL COMPARISON ==========")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        OUTPUT_DIR / "classification_metrics.csv",
        index=False
    )

    return results_df


# ==========================================
# 7. DECISION BOUNDARY VISUALIZATION
# ==========================================

def plot_decision_boundaries(data):
    # Using only two features for 2D visualization
    selected_features = [
        "petal_length",
        "petal_width"
    ]

    X = data[selected_features].values
    y = data[TARGET].values

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=RANDOM_STATE
        )
    }

    # Create Mesh Grid
    x_min = X_scaled[:, 0].min() - 0.7
    x_max = X_scaled[:, 0].max() + 0.7

    y_min = X_scaled[:, 1].min() - 0.7
    y_max = X_scaled[:, 1].max() + 0.7

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Plot
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6)
    )

    for axis, (name, model) in zip(axes, models.items()):
        model.fit(X_scaled, y)

        predictions = model.predict(grid)

        predictions = predictions.reshape(xx.shape)

        axis.contourf(
            xx,
            yy,
            pd.factorize(predictions.ravel())[0].reshape(xx.shape),
            alpha=0.3,
            cmap="viridis"
        )

        axis.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=pd.factorize(y)[0],
            cmap="viridis",
            edgecolor="black",
            s=35
        )

        axis.set_title(f"{name} Decision Boundary")

        axis.set_xlabel("Standardized Petal Length")
        axis.set_ylabel("Standardized Petal Width")

    plt.suptitle("Iris Classification Decision Boundaries")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "decision_boundaries.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

def main():
    print("Project Directory:", PROJECT_DIR)
    print("Output Directory:", OUTPUT_DIR)
    print("Output Folder Exists:", OUTPUT_DIR.exists())
    print("Files:", list(OUTPUT_DIR.iterdir()))

    print("\n")
    print("=" * 55)

    print("MULTICLASS BENCHMARKS CLASSIFICATION PIPELINE")

    print("=" * 55)

    # Load Dataset
    data = load_dataset()

    # Exploratory Data Analysis
    perform_eda(data)

    # Train Test Split
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Build Models
    models = build_models()

    # Train and Evaluate
    results = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Decision Boundary
    plot_decision_boundaries(data)

    print("\n===================================")
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("===================================")

    print("\nFinal Model Comparison:")
    print(results)

    print("\nOutput files saved in:", OUTPUT_DIR)


# ==========================================
# 9. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
