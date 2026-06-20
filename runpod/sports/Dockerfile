# RunPod serverless worker — Pythia sports-board exports.
#
# BUILD CONTEXT = REPO ROOT (so it can COPY pythia_divination/). From the repo root:
#   docker build -f runpod/sports/Dockerfile -t <registry>/pythia-sports-runpod:latest .
#   docker push  <registry>/pythia-sports-runpod:latest
# (Or point RunPod's GitHub-build at this Dockerfile path; the repo's data/ and
#  artifacts/ are gitignored so a clean clone is code-only.)
#
# The normalized sport data (pythia_divination/data/sports/**) and model artifacts
# (pythia_divination/artifacts/**) are NOT baked in — they're gitignored, kept lean
# via .dockerignore, and synced from S3 at runtime by handler.py. Code-only image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHIA_REPO=/work \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc libpq-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps: divination's full stack + the worker extras (runpod + awscli).
COPY pythia_divination/requirements.txt /tmp/requirements.txt
COPY runpod/sports/requirements.extra.txt /tmp/requirements.extra.txt
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements.extra.txt

# Divination code. data/ + artifacts/ are excluded by .dockerignore; the rm/mkdir
# below is belt-and-suspenders for local builds where the gitignored dirs exist on
# disk. The empty output dir must exist so the export scripts' PROJ_ROOT (parents[2])
# resolves to /work, matching the EC2 repo layout.
COPY pythia_divination/ /work/pythia_divination/
RUN rm -rf /work/pythia_divination/data /work/pythia_divination/artifacts \
    && mkdir -p /work/pythia_divination/data/sports \
                /work/pythia_divination/artifacts \
                /work/pythia_prophecy/frontend/src/data

COPY runpod/sports/handler.py /work/handler.py

CMD ["python", "-u", "/work/handler.py"]
