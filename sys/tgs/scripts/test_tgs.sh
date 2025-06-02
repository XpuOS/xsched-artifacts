set -o errexit
set -o pipefail
set -o nounset
set -o xtrace

pushd "$(dirname "$0")/.." >/dev/null
mkdir -p ./job_logs/exp

python3 worker.py --mount ./job_logs/exp:/workspace/espnet/egs2/aishell/asr1/exp --trace config/test_tgs.csv --log_path results/test_tgs_results.csv 2>&1 | tee backup_logs/test_tgs.log

popd >/dev/null