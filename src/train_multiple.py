import mlflow
from mlflow import sklearn as mlflow_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True)
}

def train_multiple(X_train, y_train, X_val, y_val, experiment_name="MobilePricePrediction"):
    mlflow.set_experiment(experiment_name)
    results = {}

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)

            mlflow.log_metric("accuracy", float(acc))
            mlflow_sklearn.log_model(model, artifact_path=f"{name}_model")

            results[name] = {"model": model, "accuracy": acc}

    return results