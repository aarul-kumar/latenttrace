import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from faker import Faker

from geo_utils import CITY_NAMES, pick_far_city

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
NUM_ENTITIES = 200
DAYS_OF_HISTORY = 30
EVENTS_PER_DAY_MEAN = 5
ANOMALY_RATE = 0.02  # ~2% of total events will be anomalies (spec range: 0.5-3%)

DATA_DIR = "data"
TRAIN_LOGS_PATH = os.path.join(DATA_DIR, "train_logs.csv")
STREAMING_LOGS_PATH = os.path.join(DATA_DIR, "streaming_logs.csv")

RESOURCES = ["VPN_Gateway", "HR_Portal", "Customer_DB", "Code_Repo", "Cloud_Console", "IoT_Gateway_A", "IoT_Gateway_B", "Financial_API"]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
OS_TYPES = ["Win10", "macOS", "Ubuntu20.04", "Android", "iOS", "RTOS"]
COMMAND_SEQUENCES = [
    "login_success,read,logout",
    "login_success,write,read,logout",
    "login_success,execute,logout",
    "login_success,download,logout",
    "login_failed"
]

# NOTE on labels: every attack pattern below produces its OWN dedicated
# device_fingerprint / IP / geo string. That is intentional and realistic
# (attacker infrastructure genuinely differs from user infrastructure) --
# but see profiler.py, which turns "was this fingerprint ever seen for this
# entity before" and "distance/time since the entity's last event" into
# proper numeric, behavioural features. That's what lets the detector
# generalize instead of memorizing these exact strings.


class SyntheticDataGenerator:
    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)
        np.random.seed(42)
        random.seed(42)

        self.entity_profiles = {}
        self.normal_events = []
        self.anomaly_events = []

        os.makedirs(DATA_DIR, exist_ok=True)

    def _generate_fingerprint(self, os_type: str) -> str:
        mac = self.fake.mac_address().replace(':', '-')
        return f"{os_type}_{mac}_IPv4"

    def build_entity_profiles(self):
        """Creates normal behavioral baselines for each entity."""
        for i in range(NUM_ENTITIES):
            entity_type = np.random.choice(['user', 'service_account', 'edge_device'], p=[0.7, 0.15, 0.15])

            if entity_type == 'user':
                e_id = f"U{i:04d}"
                resources = random.sample(RESOURCES, k=2)
                auth = np.random.choice(["password", "biometric"])
                os_type = np.random.choice(["Win10", "macOS", "iOS"])
            elif entity_type == 'service_account':
                e_id = f"SA{i:04d}"
                resources = ["Customer_DB", "Cloud_Console", "Financial_API"]
                auth = "token"
                os_type = "Ubuntu20.04"
            else:
                e_id = f"ED{i:04d}"
                resources = ["IoT_Gateway_A", "IoT_Gateway_B"]
                auth = "certificate"
                os_type = "RTOS"

            home_city = random.choice(CITY_NAMES)

            self.entity_profiles[e_id] = {
                "entity_type": entity_type,
                "primary_ip": self.fake.ipv4(),
                "primary_geo": home_city,
                "typical_resources": resources,
                "auth_method": auth,
                "device_fingerprint": self._generate_fingerprint(os_type),
                "typical_hour": random.randint(8, 18) if entity_type == 'user' else 0,
            }

    def generate_normal_events(self, start_date: datetime, days: int):
        """Generates benign baseline access logs."""
        print(f"Generating normal baseline events over {days} days...")
        for e_id, profile in self.entity_profiles.items():
            num_events = int(np.random.normal(EVENTS_PER_DAY_MEAN * days, 2 * days))
            num_events = max(5, num_events)

            for _ in range(num_events):
                day_offset = random.randint(0, days - 1)
                if profile['entity_type'] == 'user':
                    hour = int(np.random.normal(profile['typical_hour'], 2)) % 24
                else:
                    hour = random.randint(0, 23)

                minute = random.randint(0, 59)
                sec = random.randint(0, 59)

                event_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=sec)

                event = {
                    "entity_id": e_id,
                    "entity_type": profile["entity_type"],
                    "timestamp": event_time,
                    "source_ip": profile["primary_ip"],
                    "geo_location": profile["primary_geo"],
                    "resource_accessed": random.choice(profile["typical_resources"]),
                    "auth_method": profile["auth_method"],
                    "session_duration": abs(np.random.normal(300, 100)),
                    "command_sequence": random.choice(COMMAND_SEQUENCES[:-1]),
                    "device_fingerprint": profile["device_fingerprint"],
                    "label": "normal"
                }
                self.normal_events.append(event)

    def _inject_brute_force(self, base_time: datetime, entity_id: str):
        """Pattern 1: Rapid repeated failed-auth attempts."""
        profile = self.entity_profiles[entity_id]
        ip = self.fake.ipv4()
        geo = random.choice(CITY_NAMES)

        for i in range(15):
            self.anomaly_events.append({
                "entity_id": entity_id,
                "entity_type": profile["entity_type"],
                "timestamp": base_time + timedelta(seconds=i * 2),
                "source_ip": ip,
                "geo_location": geo,
                "resource_accessed": random.choice(profile["typical_resources"]),
                "auth_method": "password",
                "session_duration": 0.0,
                "command_sequence": "login_failed",
                "device_fingerprint": "Kali_Linux_Unknown_MAC_IPv4",
                "label": "brute_force"
            })

    def _inject_impossible_travel(self, base_time: datetime, entity_id: str):
        """Pattern 2: Distant geographic logins in an implausible timeframe.

        FIX: only the SECOND ("arrival") event is actually anomalous on its
        own merits -- the first login is completely ordinary. The original
        code labeled BOTH rows 'impossible_travel', which forced the model
        to try to learn that an entirely normal-looking row is anomalous
        (unlearnable from that row's own features, since the anomaly is a
        *relationship* between the two rows). We now label the first leg
        'normal' and let profiler.py's geo-velocity feature (distance/time
        vs the entity's OWN previous event) carry the actual signal on the
        second leg -- which is also a *varied* distant city each time,
        not a single hardcoded destination.
        """
        profile = self.entity_profiles[entity_id]
        home_city = profile["primary_geo"]
        far_city = pick_far_city(home_city, min_km=2000.0)

        # Event 1: Normal location, normal behavior -- genuinely benign.
        self.normal_events.append({
            "entity_id": entity_id,
            "entity_type": profile["entity_type"],
            "timestamp": base_time,
            "source_ip": profile["primary_ip"],
            "geo_location": home_city,
            "resource_accessed": profile["typical_resources"][0],
            "auth_method": profile["auth_method"],
            "session_duration": 120.0,
            "command_sequence": "login_success,read,logout",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "normal"
        })

        # Event 2: Arrival from a location that's thousands of km away,
        # only minutes later -- THIS is the anomalous row.
        self.anomaly_events.append({
            "entity_id": entity_id,
            "entity_type": profile["entity_type"],
            "timestamp": base_time + timedelta(minutes=10),
            "source_ip": self.fake.ipv4(),
            "geo_location": far_city,
            "resource_accessed": profile["typical_resources"][0],
            "auth_method": profile["auth_method"],
            "session_duration": 15.0,
            "command_sequence": "login_success,download,logout",
            "device_fingerprint": profile["device_fingerprint"],
            "label": "impossible_travel"
        })

    def _inject_credential_stuffing(self, base_time: datetime):
        """Pattern 3: Many entities, single source IP, high failure rate."""
        attacker_ip = self.fake.ipv4()
        attacker_geo = "Unknown, Unknown"
        targets = random.sample(list(self.entity_profiles.keys()), 30)

        for i, target_id in enumerate(targets):
            profile = self.entity_profiles[target_id]
            self.anomaly_events.append({
                "entity_id": target_id,
                "entity_type": profile["entity_type"],
                "timestamp": base_time + timedelta(seconds=i * 5),
                "source_ip": attacker_ip,
                "geo_location": attacker_geo,
                "resource_accessed": "VPN_Gateway",
                "auth_method": "password",
                "session_duration": 0.0,
                "command_sequence": "login_failed",
                "device_fingerprint": "Python_Requests_Script",
                "label": "credential_stuffing"
            })

    def _inject_lateral_movement(self, base_time: datetime, entity_id: str):
        """Pattern 4: Accessing unusual sequence of previously untouched resources."""
        profile = self.entity_profiles[entity_id]
        unusual_resources = list(set(RESOURCES) - set(profile["typical_resources"]))

        if not unusual_resources:
            return

        for i, res in enumerate(unusual_resources[:3]):
            self.anomaly_events.append({
                "entity_id": entity_id,
                "entity_type": profile["entity_type"],
                "timestamp": base_time + timedelta(minutes=i * 2),
                "source_ip": profile["primary_ip"],
                "geo_location": profile["primary_geo"],
                "resource_accessed": res,
                "auth_method": "token",
                "session_duration": 300.0,
                "command_sequence": "login_success,execute,download,logout",
                "device_fingerprint": profile["device_fingerprint"],
                "label": "lateral_movement"
            })

    def _inject_device_spoofing(self, base_time: datetime, entity_id: str):
        """Pattern 5: Matches history except mismatched device fingerprint."""
        profile = self.entity_profiles[entity_id]
        spoofed_fingerprint = self._generate_fingerprint("Unknown_OS")

        self.anomaly_events.append({
            "entity_id": entity_id,
            "entity_type": profile["entity_type"],
            "timestamp": base_time,
            "source_ip": profile["primary_ip"],
            "geo_location": profile["primary_geo"],
            "resource_accessed": profile["typical_resources"][0],
            "auth_method": profile["auth_method"],
            "session_duration": 45.0,
            "command_sequence": "login_success,read,logout",
            "device_fingerprint": spoofed_fingerprint,
            "label": "device_spoofing"
        })

    def _inject_low_and_slow(self, start_time: datetime, entity_id: str):
        """Pattern 6: Gradual, small, off-hours access over days."""
        profile = self.entity_profiles[entity_id]

        for i in range(7):
            off_hour_time = start_time + timedelta(days=i, hours=3)
            self.anomaly_events.append({
                "entity_id": entity_id,
                "entity_type": profile["entity_type"],
                "timestamp": off_hour_time,
                "source_ip": profile["primary_ip"],
                "geo_location": profile["primary_geo"],
                "resource_accessed": "Customer_DB",
                "auth_method": profile["auth_method"],
                "session_duration": 5.0,
                "command_sequence": "login_success,read,logout",
                "device_fingerprint": profile["device_fingerprint"],
                "label": "low_and_slow"
            })

    def _inject_insider_drift(self, start_time: datetime, entity_id: str):
        """Pattern 7: Slowly expanding privilege (edge case, used for FP tuning)."""
        profile = self.entity_profiles[entity_id]
        new_resource = random.choice(list(set(RESOURCES) - set(profile["typical_resources"])))

        for i in range(5):
            self.anomaly_events.append({
                "entity_id": entity_id,
                "entity_type": profile["entity_type"],
                "timestamp": start_time + timedelta(days=i * 2, hours=10),
                "source_ip": profile["primary_ip"],
                "geo_location": profile["primary_geo"],
                "resource_accessed": new_resource,
                "auth_method": profile["auth_method"],
                "session_duration": 120.0,
                "command_sequence": "login_success,read,write,logout",
                "device_fingerprint": profile["device_fingerprint"],
                "label": "insider_drift"
            })

    def inject_anomalies(self, start_date: datetime, days: int):
        """Coordinates the injection of all 7 attack patterns."""
        print("Injecting anomalous behaviors...")
        target_anomaly_count = int(len(self.normal_events) * ANOMALY_RATE)
        injections = target_anomaly_count // 7
        entities = list(self.entity_profiles.keys())

        for _ in range(max(1, injections // 15)):
            self._inject_brute_force(start_date + timedelta(days=random.randint(0, max(0, days - 1))), random.choice(entities))

        for _ in range(max(1, injections // 2)):
            self._inject_impossible_travel(start_date + timedelta(days=random.randint(0, max(0, days - 1))), random.choice(entities))

        for _ in range(max(1, injections // 30)):
            self._inject_credential_stuffing(start_date + timedelta(days=random.randint(0, max(0, days - 1))))

        for _ in range(max(1, injections // 3)):
            self._inject_lateral_movement(start_date + timedelta(days=random.randint(0, max(0, days - 1))), random.choice(entities))

        for _ in range(max(1, injections)):
            self._inject_device_spoofing(start_date + timedelta(days=random.randint(0, max(0, days - 1))), random.choice(entities))

        for _ in range(max(1, injections // 7)):
            self._inject_low_and_slow(start_date + timedelta(days=random.randint(0, max(0, days - 8))), random.choice(entities))

        for _ in range(max(1, injections // 5)):
            self._inject_insider_drift(start_date + timedelta(days=random.randint(0, max(0, days - 10))), random.choice(entities))

    def generate_datasets(self):
        """Pipeline executor to generate train and streaming datasets."""
        self.build_entity_profiles()

        train_start = datetime.now() - timedelta(days=DAYS_OF_HISTORY + 1)
        self.generate_normal_events(train_start, DAYS_OF_HISTORY)
        self.inject_anomalies(train_start, DAYS_OF_HISTORY)

        all_train_events = self.normal_events + self.anomaly_events
        df_train = pd.DataFrame(all_train_events)
        df_train.sort_values("timestamp", inplace=True)
        df_train.to_csv(TRAIN_LOGS_PATH, index=False)

        print(f"[{TRAIN_LOGS_PATH}] Generated {len(df_train)} events.")
        print("Training Label Distribution:")
        print(df_train['label'].value_counts(normalize=True).map('{:.2%}'.format))

        self.normal_events = []
        self.anomaly_events = []

        stream_start = datetime.now()
        self.generate_normal_events(stream_start, 1)
        self.inject_anomalies(stream_start, 1)

        cold_start_id = "U_COLD_999"
        self.anomaly_events.append({
            "entity_id": cold_start_id,
            "entity_type": "user",
            "timestamp": stream_start + timedelta(hours=2),
            "source_ip": self.fake.ipv4(),
            "geo_location": "Berlin, Germany",
            "resource_accessed": "Cloud_Console",
            "auth_method": "password",
            "session_duration": 200.0,
            "command_sequence": "login_success,download,logout",
            "device_fingerprint": "Ubuntu20.04_MAC_IPv4",
            "label": "cold_start_anomaly"
        })

        all_stream_events = self.normal_events + self.anomaly_events
        df_stream = pd.DataFrame(all_stream_events)
        df_stream.sort_values("timestamp", inplace=True)
        df_stream.to_csv(STREAMING_LOGS_PATH, index=False)

        print(f"\n[{STREAMING_LOGS_PATH}] Generated {len(df_stream)} events.")


if __name__ == "__main__":
    print("--- AI-Powered Behavioral Anomaly Detection ---")
    print("Initializing Data Generator...\n")
    generator = SyntheticDataGenerator()
    generator.generate_datasets()
    print("\nData Generation Complete. Ready for ML Pipeline.")