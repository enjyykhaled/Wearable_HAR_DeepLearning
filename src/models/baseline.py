import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def evaluate_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, seeds=(42, 123, 2026)):
    """Train and evaluate a Random Forest baseline across multiple random seeds."""
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    results = {"val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [], "train_time": [], "infer_time": []}
    last_preds = None

    for seed in seeds:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

        t0 = time.time()
        clf.fit(X_train_flat, y_train)
        results["train_time"].append(time.time() - t0)

        val_preds = clf.predict(X_val_flat)
        results["val_acc"].append(accuracy_score(y_val, val_preds) * 100)
        results["val_f1"].append(f1_score(y_val, val_preds, average="macro") * 100)

        t1 = time.time()
        test_preds = clf.predict(X_test_flat)
        results["infer_time"].append((time.time() - t1) / len(X_test) * 1000)

        results["test_acc"].append(accuracy_score(y_test, test_preds) * 100)
        results["test_f1"].append(f1_score(y_test, test_preds, average="macro") * 100)
        last_preds = test_preds

    return results, last_preds
    
