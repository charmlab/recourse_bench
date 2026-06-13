#!/usr/bin/env bash
#SBATCH --job-name=rb_smoke_gpu
#SBATCH --partition=JIMMY
#SBATCH --nodelist=watgpu108
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/u501/hoyarhos/recourse_bench/smoke_test/run_20260613T031401Z/gpu_retry_20260613T134000Z/slurm-%j.out
#SBATCH --error=/u501/hoyarhos/recourse_bench/smoke_test/run_20260613T031401Z/gpu_retry_20260613T134000Z/slurm-%j.err

REPO_ROOT="/u501/hoyarhos/recourse_bench"
RETRY_ROOT="$REPO_ROOT/smoke_test/run_20260613T031401Z/gpu_retry_20260613T134000Z"
PYTHON="$REPO_ROOT/.venv/bin/python"

cd "$REPO_ROOT" || exit 1

{
  echo "UTC time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Hostname: $(hostname)"
  echo "Slurm job id: ${SLURM_JOB_ID:-unknown}"
  echo "Slurm partition: ${SLURM_JOB_PARTITION:-unknown}"
  echo "Slurm GPUs: ${SLURM_JOB_GPUS:-unknown}"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Requested device: cuda, matching checked-in reference logs"
  echo "Python executable: $PYTHON"
  "$PYTHON" --version 2>&1 | sed 's/^/Python version: /'
  "$PYTHON" - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch cuda device name: {torch.cuda.get_device_name(0)}")
    print(torch.tensor([1.0], device="cuda"))
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  fi
} > "$RETRY_ROOT/environment.txt" 2>&1

for test in \
  experiment/claproar/smoke.py \
  experiment/cruds/smoke.py \
  experiment/gravitational/smoke.py \
  experiment/revise/smoke.py \
  experiment/wachter/smoke.py
do
  method="$(basename "$(dirname "$test")")"
  out="$RETRY_ROOT/${method}.out"
  err="$RETRY_ROOT/${method}.err"
  status="$RETRY_ROOT/${method}.status"
  start_epoch="$(date +%s)"
  {
    echo "METHOD=$method"
    echo "TEST=$test"
    echo "DEVICE=cuda"
    echo "COMMAND=$PYTHON $test"
    echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$status"
  "$PYTHON" "$test" > "$out" 2> "$err"
  code=$?
  end_epoch="$(date +%s)"
  {
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "DURATION_SECONDS=$((end_epoch - start_epoch))"
    echo "EXIT_CODE=$code"
    if [ "$code" -eq 0 ]; then
      echo "RESULT=PASS"
    else
      echo "RESULT=FAIL"
    fi
  } >> "$status"
  printf '%s exit=%s duration=%ss\n' "$method" "$code" "$((end_epoch - start_epoch))"
done
