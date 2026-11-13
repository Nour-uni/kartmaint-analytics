"""Synthetic data generator for a karting fleet.

IMPORTANT: this data is SIMULATED. It is not real client data (Axia Solution / any karting club).
The simulator encodes one simple, explicit assumption: parts wear with usage hours, and the
probability of a breakdown grows with hours since the part was last serviced. Preventive
maintenance is performed late/irregularly, so there is real signal (and real noise) to learn from.

Usage:
    python -m kartmaint.generate_data --karts 120 --seed 42 --sink both
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

from kartmaint.config import PARTS, PG, RAW_CSV_DIR, ROOT, SERVICE_INTERVAL_HOURS

MODELS = ["Racer A", "Racer B", "Rental C"]
INTERVAL = SERVICE_INTERVAL_HOURS
WEAR_STEEPNESS = 5.0   # how fast breakdown risk grows once a part is overdue
BASE_RATE = {"engine": 0.0002, "brakes": 0.0003, "tires": 0.00045, "chain": 0.00025}  # breakdown prob / hour at wear ratio 0.5
PART_COST = {"engine": 420.0, "brakes": 90.0, "tires": 140.0, "chain": 45.0}      # corrective cost, TND-like units
PREVENTIVE_COST_FACTOR = 0.35


def simulate(n_karts=120, start="2023-10-01", end="2026-09-30", seed=42):
    """Return clean (karts, sessions, events) DataFrames."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    start_ts = pd.Timestamp(start)
    n, n_parts = n_karts, len(PARTS)

    karts = pd.DataFrame(
        {
            "kart_id": np.arange(1, n + 1),
            "model": rng.choice(MODELS, n, p=[0.40, 0.35, 0.25]),
            "purchase_date": start_ts + pd.to_timedelta(rng.integers(-500, 120, n), unit="D"),
        }
    )
    purchase = karts["purchase_date"].values
    stress = rng.lognormal(0.0, 0.25, n)    # how hard each kart is driven
    usage = rng.lognormal(0.0, 0.30, n)     # how often each kart is rented
    interval = np.array([INTERVAL[p] for p in PARTS])
    base = np.array([BASE_RATE[p] for p in PARTS])
    cost_base = np.array([PART_COST[p] for p in PARTS])

    hours_since = rng.uniform(0.0, 0.8, (n, n_parts)) * interval   # unknown history before the data starts
    due_factor = rng.uniform(0.9, 1.5, (n, n_parts))              # how late maintenance tends to be

    s_kart, s_hours, s_day = [], [], []
    e_kart, e_part, e_type, e_cost, e_down, e_break, e_day = [], [], [], [], [], [], []

    for day in days:
        d64 = day.to_datetime64()
        active = purchase <= d64
        if not active.any():
            continue
        season = 1.0 + 0.5 * np.sin(2 * np.pi * (day.dayofyear - 100) / 365)
        weekend = 1.4 if day.weekday() >= 5 else 1.0
        n_sess = rng.poisson(1.3 * season * weekend * usage * active)
        total = int(n_sess.sum())
        hours_today = np.zeros(n)
        if total:
            kart_idx = np.repeat(np.arange(n), n_sess)
            h = np.round(rng.uniform(0.2, 0.6, total), 2)
            hours_today = np.bincount(kart_idx, weights=h, minlength=n)
            s_kart.append(kart_idx + 1)
            s_hours.append(h)
            s_day.append(np.full(total, d64))

        hours_since += hours_today[:, None]

        # 1) breakdowns: hazard rises exponentially with hours since last service
        ratio = hours_since / interval
        hazard = base * np.exp(WEAR_STEEPNESS * (ratio - 0.5)) * stress[:, None]
        p_break = 1.0 - np.exp(-hazard * hours_today[:, None])
        broke = rng.random((n, n_parts)) < p_break

        # 2) preventive maintenance: when overdue, the technician is available ~35% of days
        due = (hours_since > interval * due_factor) & active[:, None]
        prev = due & ~broke & (rng.random((n, n_parts)) < 0.35)

        for mask, kind in ((broke, "corrective"), (prev, "preventive")):
            idx = np.argwhere(mask)
            if len(idx) == 0:
                continue
            k, p = idx[:, 0], idx[:, 1]
            if kind == "corrective":
                cost = cost_base[p] * rng.lognormal(0.0, 0.4, len(idx))
                down = np.round(rng.gamma(3.0, 3.0, len(idx)) + 2.0, 1)
            else:
                cost = cost_base[p] * PREVENTIVE_COST_FACTOR * rng.lognormal(0.0, 0.2, len(idx))
                down = np.round(rng.uniform(1.0, 3.0, len(idx)), 1)
            e_kart.append(k + 1)
            e_part.append(np.array(PARTS)[p])
            e_type.append(np.full(len(idx), kind))
            e_cost.append(np.round(cost, 2))
            e_down.append(down)
            e_break.append(np.full(len(idx), 1 if kind == "corrective" else 0))
            e_day.append(np.full(len(idx), d64))

        # 3) occasional general inspections (no wear reset)
        insp = np.where((rng.random(n) < 0.004) & active)[0]
        if len(insp):
            e_kart.append(insp + 1)
            e_part.append(np.full(len(insp), "general"))
            e_type.append(np.full(len(insp), "inspection"))
            e_cost.append(np.round(rng.uniform(10, 30, len(insp)), 2))
            e_down.append(np.full(len(insp), 0.5))
            e_break.append(np.zeros(len(insp), dtype=int))
            e_day.append(np.full(len(insp), d64))

        reset = broke | prev
        hours_since[reset] = 0.0
        due_factor[reset] = rng.uniform(0.9, 1.5, int(reset.sum()))

    sessions = pd.DataFrame(
        {
            "kart_id": np.concatenate(s_kart),
            "session_date": pd.to_datetime(np.concatenate(s_day)),
            "hours_used": np.concatenate(s_hours),
        }
    ).sort_values(["session_date", "kart_id"], kind="stable").reset_index(drop=True)
    sessions.insert(0, "session_id", np.arange(1, len(sessions) + 1))

    events = pd.DataFrame(
        {
            "kart_id": np.concatenate(e_kart),
            "event_date": pd.to_datetime(np.concatenate(e_day)),
            "event_type": np.concatenate(e_type),
            "part": np.concatenate(e_part),
            "cost": np.concatenate(e_cost),
            "downtime_hours": np.concatenate(e_down),
            "is_breakdown": np.concatenate(e_break).astype(int),
        }
    ).sort_values(["event_date", "kart_id"], kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return karts, sessions, events


def inject_dirty(karts, sessions, events, rate=0.01, seed=42):
    """Add realistic data-quality problems so the ETL has real work to do."""
    rng = np.random.default_rng(seed + 1)
    s, e = sessions.copy(), events.copy()
    report = {}

    k = int(len(s) * rate * 0.3)                       # impossible session durations
    idx = rng.choice(len(s), k, replace=False)
    s.loc[s.index[idx], "hours_used"] = rng.choice([-1.0, 0.0, 25.0, 99.0], k)
    report["sessions_bad_hours"] = k

    k = int(len(e) * rate * 3)                         # inconsistent casing / whitespace
    idx = rng.choice(len(e), k, replace=False)
    fmt = [lambda v: v.upper(), lambda v: f" {v} ", lambda v: v.capitalize()]
    for i in idx:
        f = fmt[rng.integers(0, 3)]
        e.iat[i, e.columns.get_loc("part")] = f(e.iat[i, e.columns.get_loc("part")])
        if rng.random() < 0.5:
            e.iat[i, e.columns.get_loc("event_type")] = f(e.iat[i, e.columns.get_loc("event_type")])
    report["events_messy_text"] = k

    k = int(len(e) * rate * 2)                         # missing costs
    e.loc[e.index[rng.choice(len(e), k, replace=False)], "cost"] = np.nan
    report["events_null_cost"] = k

    k = int(len(e) * rate * 0.5)                       # negative downtime
    idx = e.index[rng.choice(len(e), k, replace=False)]
    e.loc[idx, "downtime_hours"] = -e.loc[idx, "downtime_hours"].abs()
    report["events_negative_downtime"] = k

    next_id = int(e["event_id"].max()) + 1
    k = int(len(events) * rate)                        # double-entry duplicates (new id, same content)
    dup = events.sample(k, random_state=seed).copy()
    dup["event_id"] = np.arange(next_id, next_id + k)
    next_id += k
    report["events_duplicates"] = k

    k = int(len(events) * rate * 0.3)                  # events dated after the data cut-off
    fut = events.sample(k, random_state=seed + 2).copy()
    fut["event_id"] = np.arange(next_id, next_id + k)
    fut["event_date"] = sessions["session_date"].max() + pd.to_timedelta(rng.integers(1, 60, k), unit="D")
    next_id += k
    report["events_future_dated"] = k

    k = int(len(events) * rate * 0.3)                  # events dated before the kart was bought
    pre = events.sample(k, random_state=seed + 3).copy()
    pre["event_id"] = np.arange(next_id, next_id + k)
    pre["event_date"] = pre["kart_id"].map(karts.set_index("kart_id")["purchase_date"]) - pd.Timedelta(days=30)
    report["events_before_purchase"] = k

    e = pd.concat([e, dup, fut, pre], ignore_index=True)
    return s, e, report


def write_csv(karts, sessions, events, out_dir: Path = RAW_CSV_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("karts", karts), ("sessions", sessions), ("maintenance_events", events)):
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def load_postgres(karts, sessions, events):
    import psycopg2  # imported lazily so tests that only use CSV don't need a database

    conn = psycopg2.connect(
        host=PG["host"], port=PG["port"], dbname=PG["db"], user=PG["user"], password=PG["password"]
    )
    with conn, conn.cursor() as cur:
        cur.execute((ROOT / "sql" / "schema.sql").read_text())
        for name, df in (("karts", karts), ("sessions", sessions), ("maintenance_events", events)):
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, date_format="%Y-%m-%d")
            buf.seek(0)
            cols = ",".join(df.columns)
            cur.copy_expert(f"COPY {name} ({cols}) FROM STDIN WITH (FORMAT csv)", buf)
    conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--karts", type=int, default=120)
    ap.add_argument("--start", default="2023-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dirty-rate", type=float, default=0.01)
    ap.add_argument("--sink", choices=["postgres", "csv", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=RAW_CSV_DIR)
    a = ap.parse_args()

    karts, sessions, events = simulate(a.karts, a.start, a.end, a.seed)
    years = (pd.Timestamp(a.end) - pd.Timestamp(a.start)).days / 365.0
    print(f"clean simulation: {len(karts)} karts, {len(sessions):,} sessions, {len(events):,} events")
    print(f"breakdowns per kart-year: {events.is_breakdown.sum() / (len(karts) * years):.2f}")

    sessions, events, report = inject_dirty(karts, sessions, events, a.dirty_rate, a.seed)
    print("injected data-quality problems:", report)

    if a.sink in ("csv", "both"):
        write_csv(karts, sessions, events, a.out_dir)
        print(f"CSV written to {a.out_dir}")
    if a.sink in ("postgres", "both"):
        load_postgres(karts, sessions, events)
        print(f"loaded into PostgreSQL at {PG['host']}:{PG['port']}/{PG['db']}")


if __name__ == "__main__":
    main()
