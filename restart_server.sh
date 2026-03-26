#!/bin/bash
# Restart the S2-Pro TTS server.
# Kills any running instance, frees GPU memory, starts fresh, waits until ready.
#
# Usage:
#   ./restart_server.sh                    # default port 8000
#   ./restart_server.sh 9000               # custom port
#   PORT=9000 ./restart_server.sh          # via env var

PORT="${1:-${PORT:-8000}}"
LOG="/tmp/sglang_server_${PORT}.log"

echo "Stopping existing server..."
ps aux | grep "sglang_omni.cli" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 5

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)
echo "GPU memory free: ${FREE_MEM:-unknown}"

echo "Starting server on port $PORT (log: $LOG)..."
source .venv/bin/activate
python -m sglang_omni.cli.cli serve --model-path fishaudio/s2-pro --port "$PORT" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

for i in $(seq 1 90); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died. Check $LOG"
        tail -20 "$LOG"
        exit 1
    fi
    if curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"running":true'; then
        echo "Ready after $((i*10))s"
        exit 0
    fi
    sleep 10
    echo "Waiting... $((i*10))s"
done

echo "Timeout after 900s. Check $LOG"
exit 1
