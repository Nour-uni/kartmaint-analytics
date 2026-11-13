#!/usr/bin/env bash
# Full pipeline with Docker (PostgreSQL + Spark inside containers). Works on Linux/macOS/Git-Bash.
set -euo pipefail
docker compose up -d db
docker compose run --rm app python -m kartmaint.generate_data --sink both
docker compose run --rm app python -m kartmaint.etl --source jdbc
docker compose run --rm app python -m kartmaint.features
docker compose run --rm app python -m kartmaint.train
docker compose run --rm app python -m kartmaint.score
echo "Done. Start the dashboard with:  docker compose up dashboard   (then open http://localhost:8501)"
