import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# -------------------------------------------------------------------------
# SHAP FEATURE ATTRIBUTION ENGINE
# -------------------------------------------------------------------------
class RiskExplainer:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model('models/detector.json')
        self.feature_cols = joblib.load('models/feature_cols.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')

        self.explainer = shap.TreeExplainer(self.model)

    @staticmethod
    def _extract_class_shap_row(shap_values, pred_class_idx: int, row_pos: int = 0) -> np.ndarray:
        """Returns the 1D per-feature SHAP array for one predicted class.

        NOTE: TreeExplainer.shap_values() for multi-class models has
        returned two different shapes across SHAP/XGBoost version
        combinations in the wild:
          (a) a LIST of length n_classes, each item an (n_samples, n_features)
              array -- the convention the original code assumed, or
          (b) a single ndarray shaped (n_samples, n_features, n_classes).
        Since this review's sandbox has no internet access to install the
        exact pinned shap==0.44.1 / xgboost==2.0.3 combination and confirm
        which one applies here, this handles BOTH shapes rather than
        assuming (a) and risking an IndexError/shape mismatch at demo time.
        """
        if isinstance(shap_values, list):
            return np.asarray(shap_values[pred_class_idx][row_pos])
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            return arr[row_pos, :, pred_class_idx]
        # binary / already-2D fallback
        return arr[row_pos]

    def explain_alert(self, features_df: pd.DataFrame) -> dict:
        """
        Generates human-readable risk factors for a specific access event.
        Requires a 1-row DataFrame of pre-processed features.
        """
        shap_values = self.explainer.shap_values(features_df)

        preds = self.model.predict_proba(features_df)
        pred_class_idx = np.argmax(preds[0])
        pred_class_name = self.label_encoder.inverse_transform([pred_class_idx])[0]
        risk_score = float(preds[0][pred_class_idx])

        if pred_class_name == "normal":
            return {
                "predicted_class": "normal",
                "risk_score": risk_score,
                "top_factors": [],
                "human_readable": "Activity fits normal behavioral baseline."
            }

        class_shap_values = self._extract_class_shap_row(shap_values, pred_class_idx, row_pos=0)

        feature_contributions = list(zip(self.feature_cols, class_shap_values))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_3 = feature_contributions[:3]
        top_factors = [{"feature": f, "impact": float(v)} for f, v in top_3]

        factor_names = [f[0].replace('_', ' ').title() for f in top_3]
        human_readable = f"Flagged as {pred_class_name.upper()} primarily due to anomalous {factor_names[0]} and {factor_names[1]}."

        return {
            "predicted_class": pred_class_name,
            "risk_score": risk_score,
            "top_factors": top_factors,
            "human_readable": human_readable,
            "shap_values_raw": class_shap_values.tolist()
        }

if __name__ == "__main__":
    print("Testing Explainability Engine...")
    try:
        explainer = RiskExplainer()
        print("SHAP Explainer initialized successfully. Ready for Dashboard integration.")
    except Exception as e:
        print(f"Error initializing Explainer. Ensure models are trained first. Error: {e}")