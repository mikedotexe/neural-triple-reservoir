#!/usr/bin/env bash
# Install reservoir launchd agents.
# Usage: ./launchd/install.sh          # install and start
#        ./launchd/install.sh --load    # install and start (same)
#        ./launchd/install.sh --unload  # stop and uninstall
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Library/LaunchAgents"
LOGDIR="$(dirname "$DIR")/logs"

AGENTS=(
    com.reservoir.service
    com.reservoir.astrid-feeder
    com.reservoir.minime-feeder
    com.reservoir.coupled-astrid
)

if [[ "${1:-}" == "--unload" ]]; then
    echo "Unloading reservoir agents..."
    for agent in "${AGENTS[@]}"; do
        if launchctl list "$agent" &>/dev/null; then
            launchctl unload "$DEST/$agent.plist" 2>/dev/null || true
            echo "  unloaded $agent"
        fi
        rm -f "$DEST/$agent.plist"
    done
    echo "Done. Agents unloaded and plists removed."
    exit 0
fi

# Stop any manually-started processes first
echo "Stopping any manually-started reservoir processes..."
pkill -f reservoir_service.py 2>/dev/null || true
pkill -f astrid_feeder.py 2>/dev/null || true
pkill -f minime_feeder.py 2>/dev/null || true
pkill -f coupled_astrid_server.py 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p "$LOGDIR"

# Install and load
echo "Installing reservoir agents to $DEST..."
mkdir -p "$DEST"
for agent in "${AGENTS[@]}"; do
    # Unload if already loaded
    if launchctl list "$agent" &>/dev/null; then
        launchctl unload "$DEST/$agent.plist" 2>/dev/null || true
    fi
    cp "$DIR/$agent.plist" "$DEST/"
    launchctl load "$DEST/$agent.plist"
    echo "  loaded $agent"
done

echo ""
echo "Done. Services starting up."
echo "  Logs: $LOGDIR/"
echo "  Status: launchctl list | grep com.reservoir"
echo "  Stop one: launchctl unload ~/Library/LaunchAgents/com.reservoir.service.plist"
echo "  Uninstall all: $DIR/install.sh --unload"
