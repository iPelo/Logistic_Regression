import pathlib
import pandas as pd

from logisctic_regression import LogisticRegressionModel
import utils


def main():
    # locate CSV (first try ./Data/wdbc.csv, then ./wdbc.csv)
    root = pathlib.Path(__file__).resolve().parent
    csv_path = root / "Data" / "wdbc.csv"
    if not csv_path.exists():
        csv_path = root / "wdbc.csv"
    data = pd.read_csv(csv_path)

    # Features (drop ID + diagnosis)
    X = data.drop(columns=["id", "diagnosis"]).values
    # Encode labels (M=1, B=0)
    y = utils.encode_binary(data["diagnosis"].values, pos_label="M")

    # Standardize & split
    X_std, mean_, std_ = utils.standardize(X)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        X_std, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Train
    model = LogisticRegressionModel(learning_rate=0.01, n_iters=5000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = utils.accuracy_score(y_test, y_pred)
    prec = utils.precision_score(y_test, y_pred)
    rec = utils.recall_score(y_test, y_pred)
    f1 = utils.f1_score(y_test, y_pred)

    print("Evaluation:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()