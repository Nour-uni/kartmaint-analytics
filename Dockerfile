FROM python:3.11-slim

# Java is required by PySpark; curl fetches the PostgreSQL JDBC driver; procps is needed by Spark scripts
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jre-headless curl procps \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/jars \
 && curl -fsSL -o /opt/jars/postgresql.jar \
    https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar

ENV PG_JAR=/opt/jars/postgresql.jar \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
