# Full pipeline with Docker on Windows PowerShell.
$ErrorActionPreference = "Stop"
docker compose up -d db
docker compose run --rm app python -m kartmaint.generate_data --sink both
docker compose run --rm app python -m kartmaint.etl --source jdbc
docker compose run --rm app python -m kartmaint.features
docker compose run --rm app python -m kartmaint.train
docker compose run --rm app python -m kartmaint.score
Write-Host "Done. Start the dashboard with:  docker compose up dashboard   (then open http://localhost:8501)"
