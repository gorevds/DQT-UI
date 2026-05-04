FROM python:3.12-slim AS base

WORKDIR /app

# System deps trimmed to runtime essentials.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY dqt/ dqt/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -e .

ENV DQT_HOST=0.0.0.0 \
    DQT_PORT=8050 \
    HOME=/tmp

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8050/ || exit 1

# Single worker to match the in-memory session store. See dqt/app/store.py.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8050", \
     "--timeout", "180", \
     "--worker-tmp-dir", "/dev/shm", \
     "dqt.app.main:server"]
