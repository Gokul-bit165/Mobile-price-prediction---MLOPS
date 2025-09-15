import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(X_train, y_train, X_val, y_val, artifacts_dir="artifacts"):
    with mlflow.start_run():
        # Model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Validation metrics
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, name="model")


        # Save locally
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(model, os.path.join(artifacts_dir, "best_model.pkl"))

    return model, acc
