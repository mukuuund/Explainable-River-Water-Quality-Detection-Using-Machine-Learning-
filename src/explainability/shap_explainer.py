"""
Phase 5 – SHAP Explainer Module
Loads the saved Phase 3.6 pipeline, computes TreeExplainer SHAP values
on transformed NWMP data, and falls back to permutation importance when needed.
"""
import os, json, logging
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────
def _feature_names_after_transform(preprocessor, raw_features):
    """Best-effort extraction of feature names after ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out(raw_features))
    except Exception:
        pass
    # fallback: inspect each transformer
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        try:
            names.extend(trans.get_feature_names_out(cols))
        except Exception:
            names.extend(cols)
    return names if names else None


# ── public API ───────────────────────────────────────────────────────
def generate_global_explanations(
    model_path: str,
    features_path: str,
    data_path: str,
    out_dir: str,
    target_col: str = "strict_compliance_label",
):
    """Return (importance_df, method_used_str)."""
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    log_path = os.path.join(out_dir, "explainer_method_log.txt")

    with open(features_path) as f:
        features = json.load(f)

    df = pd.read_csv(data_path, low_memory=False)
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    X = df[features].copy()

    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier  = pipeline[-1]                         # DecisionTreeClassifier

    # ── transform ────────────────────────────────────────────────────
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feat_names = _feature_names_after_transform(preprocessor, features) or features

    method = "Unknown"
    imp_df = pd.DataFrame()

    # ── try SHAP TreeExplainer ───────────────────────────────────────
    try:
        n_sample = min(500, X_t.shape[0])
        idx = np.random.RandomState(42).choice(X_t.shape[0], n_sample, replace=False)
        X_sample = X_t[idx]

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample)

        # multi-output → pick non-compliant class (index 1)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = (
            pd.DataFrame({"feature": feat_names[: len(mean_abs)], "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(os.path.join(out_dir, "global_shap_importance.csv"), index=False)

        # summary dot plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, feature_names=feat_names, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figures", "shap_summary_plot.png"), dpi=150)
        plt.close()

        # bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=feat_names, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figures", "shap_bar_plot.png"), dpi=150)
        plt.close()

        method = "SHAP TreeExplainer"
        log.info("SHAP TreeExplainer succeeded.")

    except Exception as e:
        log.warning(f"SHAP failed ({e}). Falling back to permutation importance.")

        from sklearn.preprocessing import LabelEncoder
        y = df[target_col].copy() if target_col in df.columns else pd.Series(pipeline.predict(X))
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))

        result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        imp_df = (
            pd.DataFrame({
                "feature": features,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        imp_df.to_csv(os.path.join(out_dir, "global_permutation_importance.csv"), index=False)

        plt.figure(figsize=(10, 6))
        top = imp_df.head(15)
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.xlabel("Permutation Importance")
        plt.title("Global Feature Importance (Permutation)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figures", "shap_bar_plot.png"), dpi=150)
        plt.close()

        method = "Permutation Importance"

    with open(log_path, "w") as f:
        f.write(f"Explanation method: {method}\n")
        f.write(f"Sample size: {n_sample if method.startswith('SHAP') else len(X)}\n")
        f.write(f"Classifier type: {type(classifier).__name__}\n")

    return imp_df, method
