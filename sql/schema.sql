-- Raw operational tables (what the kart centre's app would store).
DROP TABLE IF EXISTS maintenance_events;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS karts;

CREATE TABLE karts (
    kart_id        INT PRIMARY KEY,
    model          TEXT,
    purchase_date  DATE
);

CREATE TABLE sessions (
    session_id   BIGINT PRIMARY KEY,
    kart_id      INT REFERENCES karts(kart_id),
    session_date DATE,
    hours_used   NUMERIC(6,2)
);

-- Deliberately NOT unique on business columns: real systems get duplicates and typos.
CREATE TABLE maintenance_events (
    event_id        BIGINT PRIMARY KEY,
    kart_id         INT REFERENCES karts(kart_id),
    event_date      DATE,
    event_type      TEXT,      -- preventive | corrective | inspection
    part            TEXT,      -- engine | brakes | tires | chain | general
    cost            NUMERIC(10,2),
    downtime_hours  NUMERIC(6,2),
    is_breakdown    BOOLEAN
);

CREATE INDEX idx_sessions_kart_date ON sessions (kart_id, session_date);
CREATE INDEX idx_events_kart_date   ON maintenance_events (kart_id, event_date);
