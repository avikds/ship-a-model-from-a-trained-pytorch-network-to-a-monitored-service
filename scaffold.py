"""
Ship a Model: From a Trained PyTorch Network to a Monitored Service scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Ship a model: from a trained PyTorch network to a monitored service.

Story: train a delivery-delay model you can reproduce, package it with its preprocessing inside,
validate what comes in, save, reload and export it and prove nothing changed, batch requests,
shrink it, then watch winter traffic drift away from the training data and decide whether a
retrained model should replace it.
"""
import io
import numpy as np
import torch


def accuracy(model, X, y):
    return float(((predict_proba(model, X) >= 0.5) == (np.asarray(y) == 1)).mean())


def main():
    # ---- 1. Train a model you can reproduce ----
    X, y = load_deliveries(n=2400, seed=11)
    X_train, y_train, X_test, y_test = X[:1800], y[:1800], X[1800:], y[1800:]
    model, losses = train_model(X_train, y_train, seed=0)
    again, _ = train_model(X_train, y_train, seed=0)
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
    print("late share:", round(float(y.mean()), 3), "| loss per epoch:", [round(v, 3) for v in losses])
    print("two runs with seed 0 give identical weights:", same, "| test accuracy:", round(accuracy(model, X_test, y_test), 3))

    # ---- 2. Package it ----
    row, errors = validate_request({"distance_km": 4.2, "items": 3, "hour": 18, "driver_rating": 4.8, "traffic_index": 0.7, "rain": 1})
    print("valid request ->", row, errors)
    print("bad request ->", validate_request({"distance_km": -1, "items": 2.5, "hour": 18, "driver_rating": "high", "rain": 1, "tip": 2}))
    blob = save_artifact(model, {"version": "1.0.0", "features": FEATURES, "test_accuracy": round(accuracy(model, X_test, y_test), 4)})
    reloaded, meta = load_artifact(blob)
    print("artifact:", len(blob), "bytes | metadata:", meta)
    print("reloaded model gives identical probabilities:", bool(np.array_equal(predict_proba(model, X_test), predict_proba(reloaded, X_test))))
    exported = torch.jit.load(io.BytesIO(export_model(model, torch.as_tensor(X_test[:4]))))
    with torch.no_grad():
        diff = float((exported(torch.as_tensor(X_test)) - model(torch.as_tensor(X_test))).abs().max())
    print("TorchScript export, largest difference from the original:", diff)

    # ---- 3. Make it fast and small ----
    probs, calls = batch_predict(model, X_test.tolist(), max_batch=64)
    print("batched", len(probs), "requests in", calls, "forward passes; same answers as one call:",
          bool(np.allclose(probs, predict_proba(model, X_test), atol=1e-6)))
    small, size = quantize_weights(model, bits=8)
    print("8-bit weights:", size, "| accuracy", round(accuracy(model, X_test, y_test), 3), "->", round(accuracy(small, X_test, y_test), 3))

    # ---- 4. Watch it and roll out ----
    X_live, y_live = load_deliveries(n=1500, seed=12, winter=1.0)
    report = drift_report(X_train, X_live, FEATURES)
    print("winter drift (PSI):", report["psi"], "| drifted:", report["drifted"])
    X_recent, y_recent = load_deliveries(n=1800, seed=13, winter=1.0)
    candidate, _ = train_model(np.vstack([X_train, X_recent]), np.concatenate([y_train, y_recent]), seed=0)
    labels = np.where(np.arange(len(y_live)) % 3 == 0, y_live.astype(float), np.nan)
    shadow = shadow_compare(predict_proba(model, X_live), predict_proba(candidate, X_live), labels)
    print("shadow run on winter traffic:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shadow.items()})


if __name__ == "__main__":
    main()

