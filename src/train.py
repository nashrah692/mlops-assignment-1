import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

# dataset
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Training samples:", np.bincount(y_train))
print("Test samples:", np.bincount(y_test))

# models
models = [
    (
        "Logistic Regression",
        {"solver": "liblinear", "C": 1, "max_iter": 500},
        LogisticRegression()
    ),
    (
        "Random Forest",
        {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        RandomForestClassifier()
    ),
    (
        "XGBClassifier",
        {"use_label_encoder": False, "eval_metric": "mlogloss"},
        XGBClassifier()
    )
]

# configuring mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.set_experiment("Iris Classification")
print("Experiment info:", exp)

# model training and logging
for model_name, params, model in models:
    print(f"\n🔹 Training {model_name}...")

    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"]
        })

        cm = confusion_matrix(y_test, y_pred)
        fig_path = f"results/confusion_{model_name.replace(' ', '_')}.png"

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path, artifact_path="plots")
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
