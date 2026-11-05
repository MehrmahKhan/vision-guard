# Runs every remaining training run, one after another.
# Trimmed for CPU: 8 epochs per run, 160px throughout.
# Six runs, each differing from another in exactly one setting.
#
# Usage, from the project root with the venv active:
#     .\run_all.ps1

Write-Host "=== run2: transfer + finetune ===" -ForegroundColor Cyan
python -m src.train --model transfer --backbone MobileNetV2 --finetune --unfreeze 20 --epochs 8 --finetune-epochs 4 --run-name run2-mnv2-finetuned

Write-Host "=== run3: lower learning rate ===" -ForegroundColor Cyan
python -m src.train --model transfer --lr 1e-4 --epochs 8 --run-name run3-lr1e-4

Write-Host "=== run4: no augmentation ===" -ForegroundColor Cyan
python -m src.train --model transfer --no-augment --epochs 8 --run-name run4-no-augmentation

Write-Host "=== run5: unfreeze 40 layers ===" -ForegroundColor Cyan
python -m src.train --model transfer --finetune --unfreeze 40 --epochs 8 --finetune-epochs 4 --run-name run5-unfreeze40

Write-Host "=== run6: EfficientNetB0 backbone ===" -ForegroundColor Cyan
python -m src.train --model transfer --backbone EfficientNetB0 --epochs 8 --run-name run6-efficientnet

Write-Host "=== run7: mlflow autolog ===" -ForegroundColor Cyan
python -m src.train --model transfer --autolog --epochs 8 --run-name run7-autolog

Write-Host ""
Write-Host "All runs finished. Now run: python check_progress.py" -ForegroundColor Green
