# Closes the remaining evidence gaps. Run from C:\mlops_a3 :
#     .\evidence.ps1
# It pauses after each step with a yellow SCREENSHOT prompt.

function Banner($id, $text) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host "  SUB-TASK $id : $text" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor Cyan
    Write-Host ""
}
function Shot($name) {
    Write-Host ""
    Write-Host ">>> SCREENSHOT NOW, save as: $name" -ForegroundColor Yellow
    Read-Host ">>> press Enter when done"
}

Banner "1.1" "Environment and library versions"
python -c "import sys, tensorflow as tf, numpy, pandas, sklearn, PIL, flask, mlflow; print('python      ', sys.version.split()[0]); print('tensorflow  ', tf.__version__); print('numpy       ', numpy.__version__); print('pandas      ', pandas.__version__); print('scikit-learn', sklearn.__version__); print('pillow      ', PIL.__version__); print('flask       ', flask.__version__); print('mlflow      ', mlflow.__version__); print(''); print('GPUs visible:', tf.config.list_physical_devices('GPU')); print('(empty list is correct: tensorflow-cpu, no CUDA device)')"
Shot "1.1_environment.png"

Banner "1.2" "Dataset audit"
python -m src.audit_dataset
Shot "1.2_dataset_audit.png  (scroll up if needed, capture the counts and stats)"

Banner "2.2" "Baseline CNN model summary"
python -c "
import sys
sys.path.insert(0,'.')
from src import models, config
m = models.build_baseline_cnn()
m.summary()
"
Shot "2.2_model_summary.png"

Banner "2.7" "Keras Tuner best settings"
Get-Content artifacts\tuner_best.json
Write-Host ""
Write-Host "Trial results from the search are in artifacts\tuner\visionguard\"
Get-ChildItem artifacts\tuner\visionguard -Directory -ErrorAction SilentlyContinue | Select-Object -First 12 Name
Shot "2.7_tuner_best.png"

Banner "3.2" "Toy MLflow run against DagsHub"
python toy/toy_mlflow.py --lr 0.001 --remote
Write-Host ""
Write-Host "Now open DagsHub Experiments and find the 'toy-fashion-mnist' experiment."
Shot "3.2_toy_run_on_dagshub.png  (the terminal AND the run on DagsHub)"

Banner "3.2" "Proof that .env is git-ignored"
git check-ignore -v .env
git check-ignore -v .dvc/config.local
Shot "3.2_env_ignored.png"

Write-Host ""
Write-Host ("=" * 74) -ForegroundColor Green
Write-Host "  Quick wins done. Next: start the Flask app and capture the 422 page," -ForegroundColor Green
Write-Host "  then run .\docker_lab.ps1 warmup" -ForegroundColor Green
Write-Host ("=" * 74) -ForegroundColor Green
