# CPU-only variant. For GPU, use Dockerfile.gpu
FROM python:3.11-slim

LABEL authors="Thomas Asikis"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG USERNAME=researcher
ARG USER_UID=1001

RUN useradd -rm -d /home/${USERNAME} -s /bin/bash -u ${USER_UID} ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

# Install dependencies first (cached layer)
COPY --chown=${USERNAME} pyproject.toml uv.lock* ./
RUN uv sync --frozen --extra torch --extra jupyter 2>/dev/null || uv sync --extra torch --extra jupyter

# Copy project code
COPY --chown=${USERNAME} . .

EXPOSE 8888
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
