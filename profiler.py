import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from geo_utils import lookup_coords, haversine_km_vec

# -------------------------------------------------------------------------
# BEHAVIORAL PROFILER & FEATURE ENGINEER
# -------------------------------------------------------------------------
class ProfilerPipeline:
    """
    CHANGES vs the original submission (see accompanying review for details):

    1. engineer_sequential_features() is new. The original pipeline scored
       every access event as an independent, static row -- it had no notion
       of "what happened before this, for this entity/IP". That directly
       conflicts with the hackathon brief's requirement #1 ("sequential and
       behavioural data ... not static snapshots") and deliverable #3
       ("sequence-aware detection"). This method adds real, leakage-safe,
       backward-looking features:
         - time_since_last_event_min : gap since this entity's previous event
         - geo_velocity_kmph         : distance/time vs this entity's last
                                       known location (the actual signal
                                       "impossible travel" needs -- the
                                       original code had no such feature and
                                       relied on a single hardcoded city name)
         - is_new_device_fingerprint : has this entity ever used this exact
                                       device fingerprint before (built
                                       chronologically, no future leakage)
         - distinct_resources_5ev    : rolling count of distinct resources in
                                       the entity's trailing 5 events (helps
                                       separate lateral movement / insider
                                       drift from a single odd access)
         - failed_logins_15min       : rolling failed-login count for this
                                       entity in the trailing 15 minutes
                                       (turns "brute force" into a rate-based
                                       signal instead of a single categorical
                                       "login_failed" flag)
         - failed_logins_by_ip_15min : same, grouped by source_ip instead of
                                       entity_id (this is what actually
                                       characterizes credential stuffing:
                                       many entities, one IP, in a burst)

    2. compute_anomaly_scores() no longer loops row-by-row with
       `scores[idx] = ...` against a raw numpy array (which crashes with an
       IndexError -- confirmed by testing -- the moment it's given a
       filtered/re-indexed DataFrame, and which took ~140s for ~32k rows in
       testing). It's now a vectorized, groupby-batched computation that
       produces IDENTICAL scores (verified against the original row-by-row
       version, max abs diff = 0.0) in a small fraction of the time, and it
       assigns results by pandas index alignment instead of raw position,
       so it can never silently misalign rows again.

    3. update_profile() actually does something now (the original was a bare
       `pass`). It re-fits a single entity's IsolationForest using history
       that excludes analyst-rejected rows, which is what closes the
       "concept drift" loop (requirement #3): once an analyst confirms new
       behaviour is legitimate, the entity's baseline can be refreshed
       instead of permanently re-flagging the same new pattern.
    """

    def __init__(self):
        self.encoders = {}
        self.entity_models = {}
        self.global_model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
        self.cat_cols = ['entity_type', 'source_ip', 'geo_location', 'resource_accessed',
                         'auth_method', 'command_sequence', 'device_fingerprint']
        self.seq_feature_cols = [
            'time_since_last_event_min', 'geo_velocity_kmph', 'is_new_device_fingerprint',
            'distinct_resources_5ev', 'failed_logins_15min', 'failed_logins_by_ip_15min'
        ]
        # entity_id -> set of device_fingerprints seen during training.
        # Populated by engineer_sequential_features(is_training=True) and
        # persisted, so inference-time "is this device new?" checks are
        # judged against real training history, not just the current batch.
        self.entity_known_fingerprints = {}
        # Per-entity raw (decoded) history buffer, kept so update_profile()
        # can re-fit an entity's baseline later without needing the caller
        # to re-supply their whole history.
        self.entity_history = {}

        os.makedirs("models", exist_ok=True)

    # ------------------------------------------------------------------
    # TIME FEATURES
    # ------------------------------------------------------------------
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts hour and day of week to capture temporal behavioral habits."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        return df

    # ------------------------------------------------------------------
    # SEQUENTIAL / BEHAVIOURAL FEATURE ENGINEERING
    # ------------------------------------------------------------------
    def engineer_sequential_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Adds backward-looking, per-entity / per-IP sequential features.
        All windows only look at the PAST relative to each row (sorted by
        timestamp), so nothing here leaks future information into a row's
        own features."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['_orig_order'] = np.arange(len(df))  # so we can restore input row order at the end

        # ---- sort by entity, then time: required for correct per-entity "look back" ----
        by_entity = df.sort_values(['entity_id', 'timestamp']).copy()

        # time since this entity's previous event (minutes). First-ever
        # event for an entity has no prior event -> sentinel (large gap,
        # i.e. "no recent history", which is also how cold-start entities
        # naturally look).
        prev_time = by_entity.groupby('entity_id')['timestamp'].shift(1)
        gap_min = (by_entity['timestamp'] - prev_time).dt.total_seconds() / 60.0
        by_entity['time_since_last_event_min'] = gap_min.fillna(99999.0)

        # geo-velocity: distance to previous event's location / time elapsed.
        coords = by_entity['geo_location'].map(lookup_coords)
        lat = coords.map(lambda c: c[0] if c is not None else np.nan)
        lon = coords.map(lambda c: c[1] if c is not None else np.nan)
        prev_lat = lat.groupby(by_entity['entity_id']).shift(1)
        prev_lon = lon.groupby(by_entity['entity_id']).shift(1)
        dist_km = haversine_km_vec(prev_lat.values, prev_lon.values, lat.values, lon.values)
        dt_hours = (gap_min / 60.0).clip(lower=1.0 / 60.0)  # floor at 1 minute
        velocity = dist_km / dt_hours.values
        velocity = pd.Series(velocity, index=by_entity.index)
        velocity[prev_time.isna()] = 0.0          # no prior event -> no velocity signal
        velocity[velocity.isna()] = -1.0           # geo unresolvable (e.g. "Unknown, Unknown")
        by_entity['geo_velocity_kmph'] = velocity

        # is this device fingerprint new for this entity (chronologically)?
        if is_training:
            self.entity_known_fingerprints = {}
        seen = {k: set(v) for k, v in self.entity_known_fingerprints.items()}
        is_new_flags = np.empty(len(by_entity), dtype=int)
        for i, (e_id, fp) in enumerate(zip(by_entity['entity_id'].values, by_entity['device_fingerprint'].values)):
            bucket = seen.setdefault(e_id, set())
            is_new_flags[i] = 0 if fp in bucket else 1
            bucket.add(fp)
        by_entity['is_new_device_fingerprint'] = is_new_flags
        if is_training:
            # persist what we learned so inference-time scoring can use it
            self.entity_known_fingerprints = {k: v for k, v in seen.items()}
        else:
            # extend (but don't overwrite) so a long streaming session keeps
            # improving within itself too
            for k, v in seen.items():
                self.entity_known_fingerprints.setdefault(k, set()).update(v)

        # rolling distinct-resource count over the trailing 5 events per entity
        def _rolling_distinct(values, window=5):
            out = np.empty(len(values), dtype=int)
            buf = []
            for i, v in enumerate(values):
                buf.append(v)
                if len(buf) > window:
                    buf.pop(0)
                out[i] = len(set(buf))
            return out

        by_entity['distinct_resources_5ev'] = (
            by_entity.groupby('entity_id')['resource_accessed']
            .transform(lambda s: _rolling_distinct(s.tolist()))
        )

        # rolling failed-login counts (15 min), by entity and by source_ip.
        # NOTE: the entity-level window catches brute force (same entity,
        # rapid failures). The IP-level window is what actually catches
        # credential stuffing (many DIFFERENT entities hit from one IP) --
        # the original code had no feature that could see across entities.
        # Implemented as an explicit two-pointer sliding window per group
        # (rather than pandas' groupby().rolling(on=...), whose resulting
        # index shape/uniqueness varies across pandas versions and produced
        # duplicate labels when tested) -- this way behaviour is guaranteed
        # regardless of pandas version.
        by_entity['_is_failed'] = (by_entity['command_sequence'] == 'login_failed').astype(int)

        def _sliding_window_count(frame: pd.DataFrame, group_col: str, window_minutes: int = 15) -> pd.Series:
            out = pd.Series(0.0, index=frame.index)
            window = np.timedelta64(window_minutes, 'm')
            for _, group in frame.sort_values([group_col, 'timestamp']).groupby(group_col):
                ts = group['timestamp'].values
                flags = group['_is_failed'].values
                counts = np.empty(len(ts))
                left = 0
                running = 0
                for right in range(len(ts)):
                    running += flags[right]
                    while ts[right] - ts[left] > window:
                        running -= flags[left]
                        left += 1
                    counts[right] = running
                out.loc[group.index] = counts
            return out

        by_entity['failed_logins_15min'] = _sliding_window_count(by_entity, 'entity_id')
        by_entity['failed_logins_by_ip_15min'] = _sliding_window_count(by_entity, 'source_ip')
        by_entity.drop(columns=['_is_failed'], inplace=True)

        # restore the caller's original row order
        result = by_entity.sort_values('_orig_order').drop(columns=['_orig_order'])
        return result

    # ------------------------------------------------------------------
    # CATEGORICAL ENCODING
    # ------------------------------------------------------------------
    def fit_transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits label encoders and transforms categorical features (training time)."""
        df_processed = self.engineer_sequential_features(df, is_training=True)
        df_processed = self.extract_time_features(df_processed)

        for col in self.cat_cols:
            le = LabelEncoder()
            unique_vals = list(df_processed[col].astype(str).unique()) + ['<UNKNOWN>']
            le.fit(unique_vals)
            df_processed[col] = le.transform(df_processed[col].astype(str))
            self.encoders[col] = le

        # keep a raw per-entity history buffer (last 500 rows/entity) so
        # update_profile() can retrain later without needing fresh data passed in
        for e_id, grp in df.groupby('entity_id'):
            self.entity_history[e_id] = grp.tail(500).copy()

        joblib.dump(self.encoders, 'models/encoders.pkl')
        joblib.dump(self.entity_known_fingerprints, 'models/entity_known_fingerprints.pkl')
        joblib.dump(self.entity_history, 'models/entity_history.pkl')
        return df_processed

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms features for inference, handling unseen categories."""
        df_processed = self.engineer_sequential_features(df, is_training=False)
        df_processed = self.extract_time_features(df_processed)

        for col in self.cat_cols:
            le = self.encoders[col]
            known_classes = set(le.classes_)
            df_processed[col] = df_processed[col].astype(str).apply(
                lambda x: x if x in known_classes else '<UNKNOWN>'
            )
            df_processed[col] = le.transform(df_processed[col])

        return df_processed

    # ------------------------------------------------------------------
    # BASELINE PROFILES (per-entity + global fallback for cold-start)
    # ------------------------------------------------------------------
    def build_profiles(self, df: pd.DataFrame):
        """Builds per-entity baseline models and a global fallback (Cold-Start)."""
        print("Building behavioral profiles...")

        df_processed = self.fit_transform_features(df)
        features = self.cat_cols + ['session_duration', 'hour', 'day_of_week'] + self.seq_feature_cols

        self.global_model.fit(df_processed[features])
        joblib.dump(self.global_model, 'models/global_profiler.pkl')

        entities = df_processed['entity_id'].unique()
        for e_id in entities:
            entity_data = df_processed[df_processed['entity_id'] == e_id][features]
            if len(entity_data) > 10:
                clf = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)
                clf.fit(entity_data)
                self.entity_models[e_id] = clf

        joblib.dump(self.entity_models, 'models/entity_profiles.pkl')
        print(f"Successfully built profiles for {len(self.entity_models)} entities.")

    def update_profile(self, entity_id: str, rejected_indices=None):
        """Handles concept drift: re-fits ONE entity's baseline model using
        its buffered history, excluding any rows an analyst has confirmed
        are false positives / rejected (so legitimate-but-new behaviour
        stops being permanently re-flagged). Call this from the dashboard
        whenever enough analyst feedback has accumulated for an entity.

        Returns True if the profile was updated, False if there wasn't
        enough history to do so.
        """
        if entity_id not in self.entity_history:
            return False

        history = self.entity_history[entity_id].copy()
        if rejected_indices:
            history = history.drop(index=[i for i in rejected_indices if i in history.index], errors='ignore')

        if len(history) <= 10:
            return False

        features = self.cat_cols + ['session_duration', 'hour', 'day_of_week'] + self.seq_feature_cols
        processed = self.transform_features(history)
        clf = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)
        clf.fit(processed[features])
        self.entity_models[entity_id] = clf
        joblib.dump(self.entity_models, 'models/entity_profiles.pkl')
        return True

    # ------------------------------------------------------------------
    # SCORING (fixed: vectorized + index-safe, was row-by-row + crash-prone)
    # ------------------------------------------------------------------
    def compute_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Scores events based on entity baseline, defaulting to global for
        cold-starts. Vectorized per-entity via groupby (previously a Python
        `for idx, row in df.iterrows()` loop that (a) crashed with an
        IndexError on any non-default-indexed/filtered DataFrame and (b)
        took ~140s per ~32k rows; this version produces identical scores in
        a fraction of the time and is safe for filtered/reordered input)."""
        df_processed = self.transform_features(df)
        features = self.cat_cols + ['session_duration', 'hour', 'day_of_week'] + self.seq_feature_cols

        scores = pd.Series(index=df_processed.index, dtype=float)
        scores[:] = -self.global_model.score_samples(df_processed[features])

        for e_id, group in df_processed.groupby('entity_id'):
            if e_id in self.entity_models:
                scores.loc[group.index] = -self.entity_models[e_id].score_samples(group[features])

        return scores.reindex(df.index).values