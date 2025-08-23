import pathlib
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegressionModel
import utils

# Directory where plots will be saved
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent[1] / "notebooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_wdbc():
    """
    Locate and load the Breast Cancer Wisconsin dataset (CSV or JSON).

    The function searches in multiple likely locations relative to `src/`.
    Returns the loaded DataFrame and the path used.
    """
    root = pathlib.Path(__file__).resolve().parent
    candidates = [
        root / "data" / "wdbc.csv",
        root / "wdbc.csv",
        root.parent / "data" / "wdbc.csv",
        root.parent / "wdbc.csv",
        root / "data" / "wdbc.json",
        root / "wdbc.json",
        root.parent / "data" / "wdbc.json",
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


def plot_confusion_matrix(y_true, y_pred, name="confusion_matrix.png"):
    """
    Create and save a confusion matrix heatmap.
    """
    cm = utils.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # add numbers in each cell
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    outpath = OUTPUT_DIR / name
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved confusion matrix plot -> {outpath}")


def plot_roc_curve(y_true, y_score, name="roc_curve.png"):
    """
    Create and save an ROC curve with AUC score.
    """
    fpr, tpr, _ = utils.roc_curve(y_true, y_score)
    auc_val = utils.auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    plt.tight_layout()
    outpath = OUTPUT_DIR / name
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved ROC curve plot -> {outpath}")

def plot_loss_curve(loss_values, name="loss_curve.png"):
    """
    Create and save the training loss curve.
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_values) + 1), loss_values)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training Loss")
    plt.tight_layout()
    outpath = OUTPUT_DIR / name
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved loss curve plot -> {outpath}")


def main():
    # 1. Load dataset
    data, path = _load_wdbc()

    # 2. Extract features and labels (diagnosis -> binary encoded)
    X = data.drop(columns=["id", "diagnosis"]).values
    y = utils.encode_binary(data["diagnosis"].values, pos_label="M")

    # 3. Train/test split + standardization (fit scaler only on train set)
    X_train, X_test, y_train, y_test = utils.train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    X_train, mean_, std_ = utils.standardize(X_train)
    X_test = (X_test - mean_) / std_

    # 4. Train logistic regression model
    model = LogisticRegressionModel(learning_rate=0.01, n_iters=5000)
    model.fit(X_train, y_train)

    # 5. Evaluate model on test set
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

    # 6. Save evaluation plots
    plot_confusion_matrix(y_test, y_pred, name="confusion_matrix.png")
    proba = model.predict_proba(X_test)
    plot_roc_curve(y_test, proba, name="roc_curve.png")

    if hasattr(model, "loss_") and len(model.loss_) > 0:
        plot_loss_curve(model.loss_, name="loss_curve.png")

if __name__ == "__main__":
    main()