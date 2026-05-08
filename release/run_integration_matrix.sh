#!/bin/bash

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash release/run_integration_matrix.sh

Runs the legacy local integration matrix over iteration count, GPU mode,
torch.compile, and preload settings. This requires demo data and suitable GPU
hardware, and is not intended for CI.

Environment overrides:
  PARAMS_PATH          Params file to run.
  LOG_DIR              Directory for per-run logs.
EOF
    exit 0
fi

# Configuration Arrays
ITER_OPTS=("1" "20" "200")
GPUS=("1" "2")
COMPILE_OPTS=("1" "0")
PRELOAD_OPTS=("1" "0")
PARAMS_PATH="${PARAMS_PATH:-src/ptyrad/starter/params/examples/tBL_WSe2.yaml}"
RUN_PTYRAD_SCRIPT="./release/run_ptyrad_override.py"

LOG_DIR="${LOG_DIR:-./output/test_logs_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
START_TIME_GLOBAL=$SECONDS
FAILURES=0

echo "Starting PtyRAD Integration Test Matrix..."

for iter in "${ITER_OPTS[@]}"; do
    echo "================================================"
    echo " PHASE: $iter ITERATIONS "
    echo "================================================"
    
    for gpu in "${GPUS[@]}"; do
        for comp in "${COMPILE_OPTS[@]}"; do
            for pre in "${PRELOAD_OPTS[@]}"; do
                
                TEST_NAME="iter${iter}_gpu${gpu}_compile${comp}_preload${pre}"
                echo -n "Running $TEST_NAME... "

                # Start individual timer
                START_TEST=$SECONDS

                # Launcher logic
                if [ "$gpu" == "2" ]; then
                    # Using accelerate for multi-GPU
                    accelerate launch --multi_gpu --num_processes=2 "$RUN_PTYRAD_SCRIPT" \
                        --params_path "$PARAMS_PATH"\
                        --gpuid "acc" \
                        --n_iter "$iter" \
                        --compile "$comp" \
                        --preload "$pre" \
                        --output_path "$LOG_DIR/$TEST_NAME" \
                        > "$LOG_DIR/${TEST_NAME}.txt" 2>&1

                else
                    # Run command and capture exit code
                    python "$RUN_PTYRAD_SCRIPT" \
                        --params_path "$PARAMS_PATH"\
                        --gpuid 0 \
                        --n_iter "$iter" \
                        --compile "$comp" \
                        --preload "$pre" \
                        --output_path "$LOG_DIR/$TEST_NAME" \
                        > "$LOG_DIR/${TEST_NAME}.txt" 2>&1

                fi

                # End individual timer
                EXIT_CODE=$?
                END_TEST=$SECONDS
                DURATION=$((END_TEST - START_TEST))

                if [ $EXIT_CODE -eq 0 ]; then
                    echo "PASS ✅ (${DURATION}s)"
                else
                    echo "FAIL ❌ (${DURATION}s) - Check $LOG_DIR/${TEST_NAME}.txt"
                    FAILURES=$((FAILURES + 1))
                fi
                
            done
        done
    done
done

TOTAL_DURATION=$(($SECONDS - START_TIME_GLOBAL))
echo "================================================"
echo "All tests complete in $(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s."
if [ "$FAILURES" -gt 0 ]; then
    echo "$FAILURES matrix runs failed."
    exit 1
fi
