#!/bin/bash
# Build and run the stock-trader container with Claude Code (Max plan)
# Usage:
#   ./docker-run.sh              # Drop into bash (run claude manually)
#   ./docker-run.sh claude       # Start Claude Code directly
#
# On first run, Claude will print a URL - open it in your Mac's browser
# to authenticate with your Max plan. The token persists for the session.

set -e

IMAGE_NAME="stock-trader-dev"

echo "Building container..."
docker build -t "$IMAGE_NAME" .

echo ""
echo "Starting container..."
echo "  - Your host files are NOT mounted (full isolation)"
echo "  - Nothing inside this container can affect your Mac"
echo "  - Claude will ask you to authenticate via a browser URL on first run"
echo "  - To copy files out later: docker cp <container>:/app/. ./output/"
echo ""

if [ "$1" = "claude" ]; then
    echo "Launching Claude Code with --dangerously-skip-permissions..."
    docker run -it --rm \
        "$IMAGE_NAME" \
        claude --dangerously-skip-permissions
else
    echo "Dropping into bash. Run 'claude --dangerously-skip-permissions' when ready."
    docker run -it --rm \
        "$IMAGE_NAME"
fi
