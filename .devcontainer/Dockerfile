FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ARG PYTHON_VERSION=3.11

ENV PATH="/usr/local/bin:${PATH}"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates git build-essential gnupg; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    /root/.local/bin/uv --version; \
    install -m 0755 /root/.local/bin/uv /usr/local/bin/uv; \
    uv --version

RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get update; \
    apt-get install -y nodejs; \
    node -v; npm -v; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    npm install -g @anthropic-ai/claude-code

ENV PATH="/home/vscode/.local/bin:/usr/local/bin:${PATH}"
ENV XDG_DATA_HOME="/home/vscode/.local/share"
ENV XDG_CACHE_HOME="/home/vscode/.cache"
ENV XDG_CONFIG_HOME="/home/vscode/.config"
ENV UV_PYTHON_DIR="/home/vscode/.local/share/uv/python"

RUN set -eux; \
    mkdir -p /home/vscode/.local/bin /home/vscode/.local/share/uv /home/vscode/.cache /home/vscode/.config; \
    chown -R vscode:vscode /home/vscode/.local /home/vscode/.cache /home/vscode/.config

WORKDIR /workspaces/musubi-eval
