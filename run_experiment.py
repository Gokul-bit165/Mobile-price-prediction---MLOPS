from src.preprocessing import load_data, preprocess_data
from src.train_multiple import train_multiple
from src.evaluation import get_best_model
import mlflow

mlflow.set_experiment("MobilePricePrediction")

def main():
    # Load data
    train_df, test_df = load_data("data/raw/train.csv", "data/raw/test.csv")
    X_train, X_val, y_train, y_val, X_test, test_ids = preprocess_data(train_df, test_df)

    # Train multiple models
    results = train_multiple(X_train, y_train, X_val, y_val)

    # Get best model
    best_model = get_best_model(results)

if __name__ == "__main__":
    main()
