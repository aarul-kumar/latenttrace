# LatentTrace - AI Powered Behavioural Anomaly Detection for SOC Triage

LatentTrace is an AI-powered cybersecurity project that learns normal access behaviour for each user, service account, and edge device, then detects suspicious deviations in near real time. It classifies anomaly types, assigns a risk score, and explains why an event was flagged using SHAP-based feature attribution.

The system is built as a **SOC-style triage engine** focusing on synthetic log generation, behavioural profiling, anomaly classification, explainability, ranked alert queues, entity history timelines, analyst feedback loops, and concept-drift refreshes.

---

## Problem Statement

Modern security telemetry is sequential, noisy, and highly imbalanced. Signature-based approaches often fail to catch novel, low-and-slow, or behaviour-based attacks. 

LatentTrace addresses this by modelling **normal access behaviour over time** rather than scoring each event as an isolated snapshot. It is designed to detect complex patterns such as brute force attacks, credential stuffing, impossible travel, lateral movement, device spoofing, low-and-slow exfiltration, and insider drift.

---

## What the Project Does

LatentTrace works in four primary stages:

1. **Synthetic data generation:** Creates realistic access logs for users, service accounts, and devices, then injects multiple attack patterns.
2. **Behavioural profiling:** Learns entity-specific normal behaviour using IsolationForest, with a global fallback model for cold-start entities.
3. **Anomaly classification:** Uses XGBoost to classify the anomaly type rather than only outputting “normal” or “abnormal”.
4. **Explainable SOC triage:** Uses SHAP to explain what features contributed most to a flagged event, displaying the results in a Streamlit dashboard.

---

## Dashboard Views
<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/577b8fe9-e81a-4bb3-b4dd-ef8a0e5f302b" />

### Ranked Alert Queue
Displays the highest-risk events sorted sequentially by anomaly score.

<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/c6a4e09f-1746-4dcd-86dd-7cfc25770b50" />

### Alert Deep Dive
Exposes raw event metadata, predicted attack class, exact risk score, SHAP explanation charts, and actionable analyst feedback buttons.

<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/64d3e864-5aec-4844-ba5d-427d387f1e05" />

<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/f50676be-418b-4874-9456-4215ed855394" />

### Entity Behavioural History
Graphs the risk score timeline for a selected entity, plotting predicted labels, session durations, resource context, and the current alert threshold line for comparison.

<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/8941e004-6e63-484b-81c2-270be5ffee98" />

<img width="960" height="600" alt="image" src="https://github.com/user-attachments/assets/2aa443b0-fc41-4ec0-9a03-e88fdb01f17c" />

---

## System Architecture
<img width="1897" height="3176" alt="Blank diagram (1)" src="https://github.com/user-attachments/assets/b44dc17a-971d-479c-bf62-f1f1b0769ebb" />

---

## Key Features

* **Sequential, behavioural detection** instead of static row-level scoring
* **Per-entity profiling** for users, service accounts, and devices
* **Global fallback** for cold-start entities
* **Geo-velocity detection** for impossible travel
* **New device fingerprint detection**
* **Rolling failed-login windows** for brute force and credential stuffing
* **Distinct-resource tracking** for lateral movement
* **Multi-class attack taxonomy**
* **SHAP-based explanations** for each alert
* **Ranked top-1% alert budget**
* **Analyst feedback** and false-positive tuning
* **Entity timeline view** for behavioural history

---

## Repository Structure

```text
.
├── app.py                 # Streamlit SOC dashboard
├── data_generator.py      # Synthetic training + streaming log generator
├── detector.py            # Multi-class anomaly classifier training script
├── explainability.py      # SHAP-based explanation engine
├── profiler.py            # Behavioural profiler + sequential feature engineering
├── geo_utils.py           # City coordinates and distance utilities
├── data/
│   ├── train_logs.csv
│   ├── streaming_logs.csv
│   └── analyst_feedback.json
└── models/
    ├── detector.json
    ├── encoders.pkl
    ├── feature_cols.pkl
    ├── label_encoder.pkl
    ├── global_profiler.pkl
    ├── entity_profiles.pkl
    ├── entity_known_fingerprints.pkl
    ├── entity_history.pkl
    └── alert_threshold.pkl
```

---

## How It Works

### 1. Synthetic Data Generation
The `data_generator.py` script creates training and streaming telemetry featuring entity IDs, types, timestamps, source IPs, geo-locations, resources accessed, auth methods, session durations, command sequences, and device fingerprints. It establishes normal behaviour baselines and injects attack patterns (e.g., brute force, impossible travel, credential stuffing, lateral movement).

### 2. Behavioural Profiling
The `profiler.py` script engineers temporal and behavioural features such as time since the last event, geo-velocity, device fingerprint novelty, and rolling failed login counts. It builds a global IsolationForest for cold-start fallback and per-entity IsolationForest profiles for entities with sufficient history.

### 3. Multi-Class Detection
The `detector.py` script trains an XGBoost multi-class classifier to predict specific anomaly categories. It also computes a realistic top-1% alert budget threshold based on dynamic ranking rather than relying on arbitrary, static cutoffs.

### 4. Explainability
The `explainability.py` module utilizes a SHAP TreeExplainer to process the XGBoost output. It returns the predicted class, an overall risk score, the top contributing features driving the alert, and a human-readable SOC summary.

### 5. Dashboard
The `app.py` Streamlit dashboard brings everything together for the analyst. It features a ranked alert queue, deep-dive alert inspection, SHAP feature attribution charts, entity history timelines, cold-start visibility, and threshold filtering controls.

---

## Behaviour Patterns Simulated

* **Normal baseline:** Regular login hours, stable geo-location, typical resource usage, known device fingerprint, and ordinary command sequences.
* **Brute force:** Rapid, repeated failed authentication attempts from a single source within a short time window.
* **Impossible travel:** The same entity logging in from geographically distant cities within an implausibly short timeframe.
* **Credential stuffing:** Attempts spanning many entity IDs from a single source IP with a high failure rate.
* **Lateral movement:** Unusual sequences or breadth of resource access, including access to resources the entity has never interacted with before.
* **Device spoofing:** Normal-looking behaviour that originates from a mismatched or completely new device fingerprint.
* **Low-and-slow exfiltration:** Gradual, small-scale access during off-hours over an extended period of days.
* **Insider drift:** A slowly expanding access pattern, often treated as an edge case for false-positive tuning.

---

## Feature Engineering

LatentTrace is both sequence-aware and behaviour-aware. Instead of memorising labels, it detects anomalies using explicitly engineered feature categories:

* **Temporal features:** Hour of the day, day of the week, and time elapsed since the last event.
* **Geo features:** Geographic distance between logins, geo-velocity calculated in km/h, and impossible travel markers.
* **Device features:** Recognition of known versus new fingerprints and tracking fingerprint changes over time.
* **Sequence features:** Rolling counts of distinct resources accessed, and failed login bursts tracked by both entity and source IP.

---

## Model Stack

* **Baseline Profiler:** Scikit-learn `IsolationForest` (per entity where enough history exists, with a global fallback for unseen entities).
* **Classifier:** `XGBClassifier` (multi-class softmax probability output with class-balanced training).
* **Explainability:** SHAP `TreeExplainer` (feature-level attribution for individual alerts).
* **Dashboard Engine:** Streamlit with Plotly visualizations.

---

## Alert Budget Logic

The dashboard is designed around a **top-1% alert budget** instead of a fixed probability threshold. SOC teams do not review “all events above probability X”; they review the highest priority *N* events within a constrained budget. LatentTrace ranks all events by risk score, isolates the top 1%, and allows analysts to widen or narrow this view using a slider to perfectly align with real-world triage workflows.

---

## Analyst Feedback Loop

To reduce alert fatigue and handle concept drift, LatentTrace supports basic analyst feedback directly in the dashboard. Analysts can confirm a True Positive to record a validated incident, or flag an event as a False Positive. After accumulating enough false-positive feedback for an entity, its profile can be automatically refreshed to learn the new legitimate behaviour.

---

## Cold-Start Handling

For brand-new entities with no prior history (e.g., new employees, newly deployed devices, or sparse service accounts), LatentTrace dynamically falls back to a global model. This ensures unseen entities in streaming data are evaluated intelligently rather than causing system failures or blind guessing.

---

## Installation & Usage

**Requirements:**
* Python 3.10+ (Recommended)
* `pip`

**1. Install dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost shap streamlit plotly faker joblib
```

**2. Generate synthetic data:**
```bash
python data_generator.py
```
*(This creates `data/train_logs.csv` and `data/streaming_logs.csv`)*

**3. Train the profiler and classifier:**
```bash
python detector.py
```
*(This generates the necessary model artefacts inside the `models/` directory)*

**4. Launch the dashboard:**
```bash
streamlit run app.py
```
---

## Use Cases

* SIEM triage and alert reduction
* Authentication anomaly monitoring
* Cloud access monitoring
* Endpoint behaviour analysis
* IoT gateway security monitoring
* Privileged account and service account access analytics

---

## Research and References

* [Liu, F. T., Ting, K. M., & Zhou, Z.-H. *Isolation Forest* (2008).](https://www.lamda.nju.edu.cn/publication/icdm08b.pdf?utm_source=chatgpt.com)
* [Du, M., Li, F., Zheng, G., & Srikumar, V. *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning* (2017).](https://users.cs.utah.edu/~lifeifei/papers/deeplog.pdf?utm_source=chatgpt.com)
* [Lundberg, S. M., & Lee, S.-I. *A Unified Approach to Interpreting Model Predictions* (2017).](https://arxiv.org/abs/1705.07874?utm_source=chatgpt.com)
