# Stock Trader - Isolated Dev Container
# Run Claude Code with --dangerously-skip-permissions safely inside Docker

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libpq-dev \
    ca-certificates \
    gnupg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 (for Claude Code + frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 (ships with Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# UV package manager (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (Claude Code refuses --dangerously-skip-permissions as root)
RUN useradd -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Copy UV to dev user
RUN cp -r /root/.local /home/dev/.local && chown -R dev:dev /home/dev/.local
ENV PATH="/home/dev/.local/bin:$PATH"

# Working directory
WORKDIR /app

# Copy project files in (full isolation - no host mount)
COPY . /app

# Remove host-specific files that don't belong in the container
RUN rm -f /app/.mcp.json /app/.env

# Give dev user ownership
RUN chown -R dev:dev /app

# Git setup
RUN su dev -c 'git config --global user.email "dev@container" \
    && git config --global user.name "Container Dev" \
    && git config --global init.defaultBranch main \
    && cd /app && git init && git add -A && git commit -m "initial"'

# Switch to non-root user
USER dev

# Default: drop into a shell
CMD ["bash"]
