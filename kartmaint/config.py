"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.getenv("KART_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("KART_DATA_DIR", ROOT / "data"))

RAW_CSV_DIR = DATA_DIR / "raw_csv"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
MODEL_DIR = DATA_DIR / "model"
PRED_DIR = DATA_DIR / "predictions"

PG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "db": os.getenv("PG_DB", "kartmaint"),
    "user": os.getenv("PG_USER", "kart"),
    "password": os.getenv("PG_PASSWORD", "kart"),
}
JDBC_URL = f"jdbc:postgresql://{PG['host']}:{PG['port']}/{PG['db']}"
PG_JAR = os.getenv("PG_JAR", "/opt/jars/postgresql.jar")

PARTS = ["engine", "brakes", "tires", "chain"]
EVENT_TYPES = ["preventive", "corrective", "inspection"]
VALID_PARTS = PARTS + ["general"]
HORIZON_DAYS = 14          # we predict "breakdown in the next 14 days"
BURN_IN_DAYS = 90          # skip the first days: rolling windows are incomplete

# Manufacturer-recommended service intervals in hours of use (domain knowledge, also used by the simulator)
SERVICE_INTERVAL_HOURS = {"engine": 80.0, "brakes": 50.0, "tires": 35.0, "chain": 60.0}
