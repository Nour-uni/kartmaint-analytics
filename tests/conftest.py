import pandas as pd
import pytest

from kartmaint import etl, features, generate_data


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Run generate -> ETL -> features on a small fleet and return paths + ground truth."""
    root = tmp_path_factory.mktemp("kart")
    raw, clean, feats = root / "raw", root / "clean", root / "features"
    karts, sessions, events = generate_data.simulate(n_karts=30, start="2024-01-01", end="2024-12-31", seed=7)
    d_sessions, d_events, report = generate_data.inject_dirty(karts, sessions, events, rate=0.02, seed=7)
    generate_data.write_csv(karts, d_sessions, d_events, raw)
    etl.run("csv", raw, clean)
    features.build(clean, feats)
    return {
        "raw": raw, "clean": clean, "features": feats,
        "karts": karts, "sessions": sessions, "events": events, "report": report,
    }
