import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

MODELS = {
    "logreg": LogisticRegression(max_iter=200),
    "rf": RandomForestClassifier(n_estimators=100),
    "svm": SVC()
}

def train_multiple(X_train, y_train, X_val, y_val):
    results = {}
    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            results[name] = acc
    return results
