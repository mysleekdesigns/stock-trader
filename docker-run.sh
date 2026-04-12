#!/bin/bash
# Build and run the stock-trader container with Claude Code (Max plan)
# Usage:
#   ./docker-run.sh              # Drop into bash
#   ./docker-run.sh claude       # Start Claude Code interactively
#   ./docker-run.sh auto         # Start Claude with the PRD prompt pre-loaded
#
# On first run, Claude will print a URL - open it in your Mac's browser
# to authenticate with your Max plan.

set -e

IMAGE_NAME="stock-trader-dev"

echo "Building container..."
docker build -t "$IMAGE_NAME" .

echo ""
echo "=== Container Safety ==="
echo "  - Files are COPIED in (no host mount)"
echo "  - Nothing can affect your Mac"
echo "  - Container is ephemeral (--rm)"
echo "  - Claude will ask to authenticate via browser URL"
echo ""
echo "  To copy files out while running:"
echo "    docker ps                              # get container ID"
echo "    docker cp <id>:/app/. ./output/        # copy project out"
echo "========================"
echo ""

case "${1:-bash}" in
    claude)
        echo "Launching Claude Code (interactive)..."
        docker run -it --rm \
            --name stock-trader-dev \
            "$IMAGE_NAME" \
            claude --dangerously-skip-permissions
        ;;
    auto)
        echo "Launching Claude Code with PRD build prompt..."
        docker run -it --rm \
            --name stock-trader-dev \
            "$IMAGE_NAME" \
            claude --dangerously-skip-permissions -p "$(cat /app/start-prompt.md 2>/dev/null || cat start-prompt.md)"
        ;;
    *)
        echo "Dropping into bash. Run 'claude --dangerously-skip-permissions' when ready."
        docker run -it --rm \
            --name stock-trader-dev \
            "$IMAGE_NAME"
        ;;
esac
