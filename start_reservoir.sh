#!/usr/bin/env bash
# Start the ANE triple reservoir service and entity feeders.
#
# Starts 3 processes:
#   1. reservoir_service.py  — WebSocket server on port 7881
#   2. astrid_feeder.py      — polls bridge.db, feeds astrid + claude_main handles
#   3. minime_feeder.py      — polls spectral_state.json, feeds minime + claude_main handles
#
# Stop with: pkill -f reservoir_service; pkill -f astrid_feeder; pkill -f minime_feeder

set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate

mkdir -p state

echo "[reservoir] starting service on port 7881..."
python -u reservoir_service.py --port 7881 --state-dir state/ &
sleep 2

echo "[reservoir] starting Astrid feeder..."
python -u astrid_feeder.py &

echo "[reservoir] starting minime feeder..."
python -u minime_feeder.py &

echo "[reservoir] all processes started."
echo "  PIDs: service=$!, feeders running in background"
echo "  Stop: pkill -f reservoir_service; pkill -f astrid_feeder; pkill -f minime_feeder"

wait
