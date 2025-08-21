import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from logisctic_regression import LogisticRegressionModel
import utils

OUTPUT_DIR = pathlib.Path("/Users/r.ilay/Desktop/AI Portfolio/Breast Cancer Classifier/notebooks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_wdbc():
    root = pathlib.Path(__file__).resolve().parent  # e.g., .../src
    candidates = [
        root / "Data" / "wdbc.csv",
        root / "wdbc.csv",
        root.parent / "Data" / "wdbc.csv",
        root.parent / "wdbc.csv",
        root / "Data" / "wdbc.json",
        root / "wdbc.json",
        root.parent / "Data" / "wdbc.json",
        root.parent / "wdbc.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("wdbc dataset not found at any of:\n" + "\n".join(map(str, candidates)))
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path)
    return df, path


def main():
    data, path = _load_wdbc()
    # features/labels
    X = data.drop(columns=["id", "diagnosis"]).values
    y = utils.encode_binary(data["diagnosis"].values, pos_label="M")

    # preprocess
    X_std, mean_, std_ = utils.standardize(X)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        X_std, y, test_size=0.2, shuffle=True, random_state=42
    )

    # train
    model = LogisticRegressionModel(learning_rate=0.01, n_iters=5000)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    acc = utils.accuracy_score(y_test, y_pred)
    prec = utils.precision_score(y_test, y_pred)
    rec = utils.recall_score(y_test, y_pred)
    f1 = utils.f1_score(y_test, y_pred)

    print(f"Loaded dataset from: {path}")
    print("Evaluation:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    # plots
    plot_confusion_matrix(y_test, y_pred, name="confusion_matrix.png")
    proba = model.predict_proba(X_test)
    plot_roc_curve(y_test, proba, name="roc_curve.png")


def plot_confusion_matrix(y_true, y_pred, name="confusion_matrix.png"):
    cm = utils.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    outpath = OUTPUT_DIR / name
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {outpath}")


def plot_roc_curve(y_true, y_score, name="roc_curve.png"):
    fpr, tpr, _ = utils.roc_curve(y_true, y_score)
    auc_val = utils.auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    plt.tight_layout()
    outpath = OUTPUT_DIR / name
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved ROC curve plot -> {outpath}")


if __name__ == "__main__":
    main()