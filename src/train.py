import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def make_data(n=500, seed=42):
    """Generate a synthetic house price dataset."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(500, 3500, n)
    bedrooms = rng.integers(1, 6, n)
    age = rng.uniform(0, 40, n)
    price = 50 * area + 10000 * bedrooms - 800 * age + rng.normal(0, 15000, n)
    return pd.DataFrame({"area": area, "bedrooms": bedrooms, "age": age, "price": price})


def main():
    df = make_data()
    X = df[["area", "bedrooms", "age"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
