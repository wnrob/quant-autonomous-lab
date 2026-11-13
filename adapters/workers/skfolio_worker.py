"""skfolio worker (runs inside envs/portfolio). Reads a .npy matrix of daily returns, fits the requested optimiser on each [start, end) row window,
prints one JSON line {"weights": [[...], ...]} (one weight vector per window). Long-only, fully invested, per-asset cap = max_weight."""
import json, sys
import numpy as np

def main(a: dict) -> dict:
    from skfolio.optimization import HierarchicalRiskParity, MeanRisk, ObjectiveFunction
    X = np.load(a["data"]); n = X.shape[1]; cap = float(a["max_weight"]); method = a["method"]
    if cap * n < 1 - 1e-9: raise ValueError(f"max_weight {cap} infeasible for {n} assets")
    out = []
    for s, e in a["windows"]:
        import pandas as pd
        R = pd.DataFrame(X[s:e])
        if method == "hrp": m = HierarchicalRiskParity(max_weights=cap, min_weights=0.0)
        elif method == "min_variance": m = MeanRisk(objective_function=ObjectiveFunction.MINIMIZE_RISK, min_weights=0.0, max_weights=cap)
        elif method == "mean_risk": m = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, min_weights=0.0, max_weights=cap)
        else: raise ValueError(f"unknown method {method}")
        m.fit(R); w = np.asarray(m.weights_, dtype=float); w = np.clip(w, 0, None); out.append((w / w.sum()).tolist())
    return {"weights": out}

if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
