import numpy as np

# ======================================================
# Data preprocessing utilities
# ======================================================

def encode_binary(y, pos_label="M"):
    """
    Encode labels into binary {0,1}.
    Args:
        y : array-like of labels
        pos_label : the label considered as positive (mapped to 1)
    Returns:
        numpy array of 0/1
    """
    y = np.asarray(y)
    return (y == pos_label).astype(int)


def standardize(X, mean_=None, std_=None, eps=1e-8):
    """
    Standardize features to mean=0 and std=1.
    If mean/std are provided, apply them (useful for test data).
    """
    X = np.asarray(X, dtype=float)
    if mean_ is None:
        mean_ = X.mean(axis=0)
    if std_ is None:
        std_ = X.std(axis=0)
    # avoid division by zero
    std_ = np.where(std_ < eps, 1.0, std_)
    return (X - mean_) / std_, mean_, std_


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Simple implementation of train/test split.
    Args:
        X : features
        y : labels
        test_size : fraction for test set
        shuffle : whether to shuffle before splitting
        random_state : reproducible seed
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    cut = int(n * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


# ======================================================
# Prediction utilities
# ======================================================

def predict_labels(proba, threshold=0.5):
    """
    Convert probabilities into binary class predictions.
    """
    proba = np.asarray(proba)
    return (proba >= threshold).astype(int)


# ======================================================
# Evaluation metrics
# ======================================================

def confusion_matrix(y_true, y_pred):
    """
    Compute a 2x2 confusion matrix:
        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def accuracy_score(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, eps=1e-12):
    """TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    return tp / (tp + fp + eps)


def recall_score(y_true, y_pred, eps=1e-12):
    """TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    return tp / (tp + fn + eps)


def f1_score(y_true, y_pred, eps=1e-12):
    """Harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred, eps)
    r = recall_score(y_true, y_pred, eps)
    return 2 * p * r / (p + r + eps)


# ======================================================
# ROC / AUC utilities
# ======================================================

def roc_curve(y_true, y_score):
    """
    Compute ROC curve points.
    Args:
        y_true : ground truth labels {0,1}
        y_score : predicted probabilities
    Returns:
        fpr, tpr, thresholds (thresholds not used here)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # sort by score (descending)
    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / (P if P > 0 else 1)
    fpr = fps / (N if N > 0 else 1)

    # prepend (0,0) and append (1,1)
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return fpr, tpr, None


def auc(fpr, tpr):
    """Area under ROC curve (trapezoidal rule)."""
    return np.trapz(tpr, fpr)