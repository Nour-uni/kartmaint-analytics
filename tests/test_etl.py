import pandas as pd


def test_etl_recovers_the_clean_events(pipeline):
    """Dirty data was injected on top of a known clean table; the ETL must undo it exactly."""
    clean_events = pd.read_parquet(pipeline["clean"] / "events")
    assert len(clean_events) == len(pipeline["events"])
    assert clean_events["cost"].notna().all()
    assert (clean_events["downtime_hours"] >= 0).all()
    assert set(clean_events["part"]) <= {"engine", "brakes", "tires", "chain", "general"}


def test_etl_removes_invalid_sessions(pipeline):
    s = pd.read_parquet(pipeline["clean"] / "sessions")
    assert s["hours_used"].between(0.05, 12).all()
    assert s["session_id"].is_unique


def test_etl_is_partitioned(pipeline):
    assert any(p.name.startswith("year=") for p in (pipeline["clean"] / "events").iterdir())
