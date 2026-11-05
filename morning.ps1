# Finishes Task 2.7 and Task 3 while you are at class.
# Usage from the project root with the venv active:
#     .\morning.ps1

Write-Host "=== Keras Tuner, 10 trials (2.7) ===" -ForegroundColor Cyan
python -m src.tune --trials 10 --epochs 4

Write-Host "=== Run comparison table (3.5) ===" -ForegroundColor Cyan
python -m src.compare_runs

Write-Host "=== Register champion (3.6) ===" -ForegroundColor Cyan
python pick_champion.py

Write-Host ""
Write-Host "Done. Run: python check_progress.py" -ForegroundColor Green
