#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG_ARG="${1:-configs/ollama_test_gemini_2s_8t.json}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"
BASE_SLEEP="${BASE_SLEEP:-60}"
MAX_SLEEP="${MAX_SLEEP:-900}"
SUPERVISOR_LOG_DIR="${SUPERVISOR_LOG_DIR:-$SCRIPT_DIR/supervisor_logs}"
SUPERVISOR_TEE="${SUPERVISOR_TEE:-1}"

if [[ -t 1 && -z "${NO_COLOR:-}" && -z "${FORCE_COLOR:-}" ]]; then
    export FORCE_COLOR=1
fi

trap 'echo "[$(date --iso-8601=seconds)] Interrupted by user."; exit 130' INT TERM

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$CONFIG_ARG" = /* ]]; then
    CONFIG_PATH="$CONFIG_ARG"
elif [[ -f "$PWD/$CONFIG_ARG" ]]; then
    CONFIG_PATH="$PWD/$CONFIG_ARG"
elif [[ -f "$SCRIPT_DIR/$CONFIG_ARG" ]]; then
    CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
else
    echo "Config file not found: $CONFIG_ARG" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$SUPERVISOR_LOG_DIR"
cd "$SCRIPT_DIR"

attempt=1
while (( attempt <= MAX_RESTARTS )); do
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_file="$SUPERVISOR_LOG_DIR/run_${timestamp}_attempt_${attempt}.log"

    echo "[$(date --iso-8601=seconds)] Starting attempt $attempt/$MAX_RESTARTS"
    echo "[$(date --iso-8601=seconds)] Config: $CONFIG_PATH"

    if [[ "$SUPERVISOR_TEE" == "0" ]]; then
        "$PYTHON_BIN" simulation_runner.py --config "$CONFIG_PATH"
        exit_code=$?
    else
        "$PYTHON_BIN" simulation_runner.py --config "$CONFIG_PATH" 2>&1 | tee "$log_file"
        exit_code=${PIPESTATUS[0]}

        if grep -q -- "--- Simulation Complete ---" "$log_file"; then
            echo "[$(date --iso-8601=seconds)] Simulation completed successfully."
            exit 0
        fi

        if grep -q -- "Simulation was already complete" "$log_file"; then
            echo "[$(date --iso-8601=seconds)] Simulation was already complete."
            exit 0
        fi
    fi

    if (( exit_code == 0 )); then
        echo "[$(date --iso-8601=seconds)] Simulation completed successfully."
        exit 0
    fi

    if [[ "$SUPERVISOR_TEE" == "0" ]]; then
        log_ref="no per-attempt log because SUPERVISOR_TEE=0"
    else
        log_ref="$log_file"
    fi

    if (( exit_code == 1 )); then
        echo "[$(date --iso-8601=seconds)] Fatal error. Stopping supervisor. Log: $log_ref" >&2
        exit "$exit_code"
    fi

    if (( exit_code == 120 || exit_code == 130 || exit_code == 143 )); then
        echo "[$(date --iso-8601=seconds)] Interrupted. Stopping supervisor. Log: $log_ref" >&2
        exit "$exit_code"
    fi

    sleep_s=$(( BASE_SLEEP * attempt ))
    if (( sleep_s > MAX_SLEEP )); then
        sleep_s="$MAX_SLEEP"
    fi

    echo "[$(date --iso-8601=seconds)] Recoverable or ambiguous failure with exit code $exit_code."
    echo "[$(date --iso-8601=seconds)] Sleeping ${sleep_s}s before restart. Log: $log_ref"
    sleep "$sleep_s"

    attempt=$(( attempt + 1 ))
done

echo "[$(date --iso-8601=seconds)] Maximum restarts reached without completion." >&2
exit 2
