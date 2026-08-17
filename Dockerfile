# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.8.14 AS uv

FROM vllm/vllm-openai:nightly

# CRIU is available through the Ubuntu PPA. cuda-checkpoint is built and
# published by NVIDIA as a standalone utility for CUDA process checkpointing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common \
      git \
      iptables \
 && add-apt-repository -y ppa:criu/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends criu \
 && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint \
 && install -m 0755 \
      /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint \
      /usr/local/bin/cuda-checkpoint \
 && rm -rf /tmp/cuda-checkpoint /var/lib/apt/lists/*

# Keep the launcher and its runtime patches in an isolated uv project.
WORKDIR /opt/vllm-project

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY tests ./tests

RUN uv sync --locked --no-dev --no-cache

ENTRYPOINT ["uv", "run", "--project", "/opt/vllm-project", "--no-sync", "python", "-m", "vllm_criu.launcher"]
