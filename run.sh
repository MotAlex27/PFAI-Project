#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="$PYTHON"
elif [[ -x "$HOME/.venvs/traffic-cityflow/bin/python" ]]; then
    PYTHON="$HOME/.venvs/traffic-cityflow/bin/python"
else
    PYTHON="python3"
fi
EPISODES="${EPISODES:-100}"
MAX_SECONDS="${MAX_SECONDS:-1800}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"

usage() {
    cat <<'EOF'
Usage: bash run.sh <command> [options]

Commands:
  check       Check Python dependencies and CUDA availability
  generate    Generate the CityFlow corridor scenario
  test        Run the project unit tests
  train       Train both DQN agents (CUDA required)
  demo        Run DQN inference and export a CityFlow replay
  evaluate    Compare fixed-time, Max-Pressure, and DQN
  all         Generate, test, train, run the demo, and evaluate
  help        Show this help

Options after a command are passed to its Python script.

Environment overrides:
  PYTHON         Python executable (default: python3)
  EPISODES       Episodes for 'all' (default: 100)
  MAX_SECONDS    Episode duration for 'all' (default: 1800)
  EVAL_EPISODES  Evaluation seeds for 'all' (default: 5)

Before training, install a CUDA-enabled PyTorch build in the selected Python
environment, then install this project with: python -m pip install -e .
EOF
}

run_python() {
    "$PYTHON" "$@"
}

check_dependencies() {
    run_python -c '
import importlib.util
import sys

required = ("cityflow", "gymnasium", "matplotlib", "numpy", "torch")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install project dependencies with: python -m pip install -e .", file=sys.stderr)
    raise SystemExit(1)
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'
}

require_cuda() {
    run_python -c '
import sys
import torch
if not torch.cuda.is_available():
    print(
        "Training requires CUDA, but this Python environment cannot access a GPU. "
        "Install CUDA-enabled PyTorch and verify WSL2 GPU access.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Training on {torch.cuda.get_device_name(0)}")
'
}

cd "$PROJECT_DIR"

command="${1:-help}"
if (($# > 0)); then
    shift
fi

case "$command" in
    check)
        check_dependencies
        ;;
    generate)
        run_python generate_scenario.py "$@"
        ;;
    test)
        PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
            run_python -m unittest discover -s tests -v "$@"
        ;;
    train)
        check_dependencies
        require_cuda
        run_python train.py "$@"
        ;;
    demo)
        check_dependencies
        run_python demo.py "$@"
        ;;
    evaluate)
        check_dependencies
        run_python evaluate.py "$@"
        ;;
    all)
        check_dependencies
        require_cuda
        run_python generate_scenario.py
        PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
            run_python -m unittest discover -s tests -v
        run_python train.py \
            --episodes "$EPISODES" \
            --max-seconds "$MAX_SECONDS"
        run_python demo.py
        run_python evaluate.py \
            --episodes "$EVAL_EPISODES" \
            --max-seconds "$MAX_SECONDS"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
