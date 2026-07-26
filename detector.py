import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from profiler import ProfilerPipeline

# -------------------------------------------------------------------------
# MULTI-CLASS ANOMALY DETECTOR
# -------------------------------------------------------------------------
class AnomalyDetector:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            # NOTE: `use_label_encoder` was removed from XGBoost's sklearn API
            # in 1.6+ (it does nothing in the xgboost==2.0.3 this project
            # pins) -- passing it raises a TypeError at construction time,
            # before training even starts. Label encoding is already handled
            # explicitly below via self.label_encoder, so it isn't needed.
            max_depth=6,
            learning_rate=0.1,
            n_estimators=150,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.alert_threshold = None  # set by optimize_precision_at_k after training

    def prepare_training_data(self, df_raw: pd.DataFrame, profiler: ProfilerPipeline):
        """Merges profiler scores with raw + engineered features for the supervised model."""
        df = profiler.fit_transform_features(df_raw)
        df['baseline_anomaly_score'] = profiler.compute_anomaly_scores(df_raw)

        self.feature_cols = (
            profiler.cat_cols
            + ['session_duration', 'hour', 'day_of_week']
            + profiler.seq_feature_cols
            + ['baseline_anomaly_score']
        )

        X = df[self.feature_cols]
        y = self.label_encoder.fit_transform(df['label'])
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.feature_cols, 'models/feature_cols.pkl')

        return X, y

    def optimize_precision_at_k(self, y_true, y_probs, k_fraction=0.01):
        """Finds the score threshold that corresponds to the top-k_fraction
        most-anomalous events (the 'realistic analyst alert budget' from the
        evaluation criteria), and reports precision/recall AT that budget.
        NOTE: in the original submission this method existed but was never
        called anywhere -- the dashboard's "Top 1% Alert Budget" heading
        wasn't backed by any actual top-1% computation, just a user-movable
        0.50 probability slider. This wires it up for real and persists the
        resulting threshold so the dashboard can use it as a principled
        default instead of an arbitrary number.
        """
        normal_idx = self.label_encoder.transform(['normal'])[0]
        anomaly_probs = 1.0 - y_probs[:, normal_idx]
        threshold = np.percentile(anomaly_probs, 100 * (1.0 - k_fraction))

        flagged = anomaly_probs >= threshold
        actually_anomalous = (y_true != normal_idx)
        if flagged.sum() > 0:
            precision_at_k = (flagged & actually_anomalous).sum() / flagged.sum()
        else:
            precision_at_k = float('nan')
        recall_at_k = (flagged & actually_anomalous).sum() / max(1, actually_anomalous.sum())

        print(f"\n--- Precision-at-K (top {k_fraction:.1%} alert budget) ---")
        print(f"Score threshold:  {threshold:.4f}")
        print(f"Alerts raised:    {flagged.sum()} / {len(y_true)} events ({flagged.mean():.2%})")
        print(f"Precision@K:      {precision_at_k:.2%}  (of alerts raised, % that are true anomalies)")
        print(f"Recall@K:         {recall_at_k:.2%}  (of all true anomalies, % caught within the budget)")

        self.alert_threshold = float(threshold)
        joblib.dump(self.alert_threshold, 'models/alert_threshold.pkl')
        return threshold

    def train(self, X: pd.DataFrame, y: np.ndarray):
        print("Training Multi-Class XGBoost Detector...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        weights = compute_sample_weight('balanced', y_train)

        self.model.fit(X_train, y_train, sample_weight=weights)

        preds = self.model.predict(X_test)
        target_names = self.label_encoder.classes_
        print("\n--- Classification Report (Validation) ---")
        print(classification_report(y_test, preds, target_names=target_names))

        probs_test = self.model.predict_proba(X_test)
        self.optimize_precision_at_k(y_test, probs_test, k_fraction=0.01)

        self.model.save_model('models/detector.json')
        print("\nModel saved to models/detector.json")


# -------------------------------------------------------------------------
# PIPELINE EXECUTION SCRIPT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading synthetic training data...")
    train_path = "data/train_logs.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError("Please run data_generator.py first.")

    df_train = pd.read_csv(train_path)

    # 1. Initialize and run Profiler
    profiler = ProfilerPipeline()
    profiler.build_profiles(df_train)

    # 2. Initialize and run Detector
    detector = AnomalyDetector()
    X, y = detector.prepare_training_data(df_train, profiler)
    detector.train(X, y)

    print("Machine Learning Pipeline completed successfully.")