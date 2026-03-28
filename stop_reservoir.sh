#!/usr/bin/env bash
# Stop the reservoir service and feeders gracefully.
echo "[reservoir] stopping feeders..."
pkill -f astrid_feeder.py 2>/dev/null || true
pkill -f minime_feeder.py 2>/dev/null || true
sleep 1
echo "[reservoir] stopping service..."
pkill -f reservoir_service.py 2>/dev/null || true
sleep 2
echo "[reservoir] all stopped."
