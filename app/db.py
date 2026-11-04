"""Task 5 - the database trail.

One table, predictions. Every request that reaches the model writes exactly one
row: what was decided, when, by which model, and how confidently.

DATABASE_URL picks the backend:
    sqlite:///visionguard.db                      -> local SQLite file
    postgresql://visionguard:<password>@db:5432/visionguard -> Postgres in Compose

No ORM, just the two drivers behind one small interface, because the schema is a
single table and an ORM would be more moving parts than the task needs.
"""

import os
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlparse

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    file_name       TEXT    NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    verdict         TEXT    NOT NULL,
    probability_ai  REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    threshold       REAL    NOT NULL,
    model_version   TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    latency_ms      INTEGER NOT NULL,
    client_ip       TEXT
);
"""

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ      NOT NULL,
    file_name       TEXT             NOT NULL,
    file_size_bytes BIGINT           NOT NULL,
    verdict         TEXT             NOT NULL,
    probability_ai  DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION NOT NULL,
    model_version   TEXT             NOT NULL,
    source          TEXT             NOT NULL,
    latency_ms      BIGINT           NOT NULL,
    client_ip       TEXT
);
"""


def database_url():
    return os.environ.get("DATABASE_URL", "sqlite:///visionguard.db")


def is_postgres():
    url = database_url()
    return url.startswith("postgresql://") or url.startswith("postgres://")


def _sqlite_path():
    url = database_url()
    # sqlite:///visionguard.db  ->  visionguard.db
    return url.replace("sqlite:///", "", 1)


def get_connection():
    if is_postgres():
        import psycopg2

        parsed = urlparse(database_url())
        port = parsed.port
        if port is None:
            port = 5432
        return psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=port,
        )

    connection = sqlite3.connect(_sqlite_path())
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()
    if is_postgres():
        cursor.execute(POSTGRES_SCHEMA)
    else:
        cursor.execute(SQLITE_SCHEMA)
    connection.commit()
    cursor.close()
    connection.close()
    print("database ready: " + database_url().split("@")[-1])


def log_prediction(
    file_name,
    file_size_bytes,
    result,
    source,
    latency_ms,
    client_ip=None,
):
    connection = get_connection()
    cursor = connection.cursor()

    now = datetime.now(timezone.utc)
    if is_postgres():
        statement = (
            "INSERT INTO predictions (created_at, file_name, file_size_bytes, verdict, "
            "probability_ai, confidence, threshold, model_version, source, latency_ms, "
            "client_ip) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        )
        timestamp = now
    else:
        statement = (
            "INSERT INTO predictions (created_at, file_name, file_size_bytes, verdict, "
            "probability_ai, confidence, threshold, model_version, source, latency_ms, "
            "client_ip) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        )
        timestamp = now.isoformat()

    values = (
        timestamp,
        file_name,
        int(file_size_bytes),
        result["verdict"],
        float(result["probability_ai"]),
        float(result["confidence"]),
        float(result["threshold"]),
        result["model"],
        source,
        int(latency_ms),
        client_ip,
    )

    cursor.execute(statement, values)
    connection.commit()
    cursor.close()
    connection.close()


def fetch_history(limit=100):
    connection = get_connection()
    cursor = connection.cursor()

    if is_postgres():
        cursor.execute(
            "SELECT id, created_at, file_name, file_size_bytes, verdict, probability_ai, "
            "confidence, threshold, model_version, source, latency_ms FROM predictions "
            "ORDER BY id DESC LIMIT %s",
            (limit,),
        )
    else:
        cursor.execute(
            "SELECT id, created_at, file_name, file_size_bytes, verdict, probability_ai, "
            "confidence, threshold, model_version, source, latency_ms FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )

    column_names = []
    for description in cursor.description:
        column_names.append(description[0])

    rows = []
    for raw_row in cursor.fetchall():
        record = {}
        index = 0
        while index < len(column_names):
            record[column_names[index]] = raw_row[index]
            index = index + 1
        rows.append(record)

    cursor.close()
    connection.close()
    return rows


def summary_counts():
    rows = fetch_history(limit=100000)
    totals = {}
    totals["total"] = len(rows)
    totals["ai"] = 0
    totals["real"] = 0
    for row in rows:
        if row["verdict"] == "AI":
            totals["ai"] = totals["ai"] + 1
        else:
            totals["real"] = totals["real"] + 1
    return totals


if __name__ == "__main__":
    init_db()
