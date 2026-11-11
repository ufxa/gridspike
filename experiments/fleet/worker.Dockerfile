# FedGrid Vast.ai worker image (the fleet module deliverable 1).
#
# Pins a cu12.x PyTorch 2.x base so the same image boots on every snapshot host
# (driver 535..595; VAST_BUDGET §3) -- a cu13 image would exclude ~19% of the book.
# The fedgrid wheel + experiments package are baked in; the ~85 MB dataset is NOT
# (it is fetched + sha256-verified at onstart, keeping the image slim per §1/§4).
#
# Build context = repo root, with the wheel already built into ./dist:
#   ./.venv/Scripts/python.exe -m build --wheel        # -> dist/fedgrid-*.whl
#   docker build -f experiments/fleet/worker.Dockerfile \
#       -t <yourrepo>/fedgrid-worker:<tag> .
#   docker push <yourrepo>/fedgrid-worker:<tag>        # public tag; NO secrets baked
#
# Re-confirm the base tag against the repo's torch pin on launch day (§7.3).
#
# The base MUST ship an sshd. Vast's SSH launch mode replaces the image ENTRYPOINT
# with its own setup script and only *starts* a daemon the image already has, so a
# bare `pytorch/pytorch:*` base is unreachable on port 22 (every `execute`/`copy`
# then fails) -- that cost a full paid rehearsal on 2026-07-28. `vastai/pytorch`
# ships openssh + Vast's instance-tools and is the verified base.
ARG BASE_IMAGE=vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15
FROM ${BASE_IMAGE}

# curl + ca-certificates for the onstart dataset fetch; git for provenance capture.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ENV FEDGRID_BUNDLE=/workspace/bundle \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/bundle
WORKDIR /workspace

# Bake the wheel + experiments package + worker scripts into the bundle dir. The
# controller overlays manifest.jsonl (and, if needed, a fresher wheel) at run
# time via `vastai copy`, so image and battery definition stay decoupled.
COPY dist/*.whl /workspace/bundle/
COPY experiments /workspace/bundle/experiments
COPY experiments/fleet/onstart.sh experiments/fleet/run_shard.sh /workspace/bundle/

# torch is already provided by the base at the pinned cu12.x build; installing the
# wheel pulls only the remaining runtime deps and never downgrades torch.
RUN pip install --no-cache-dir /workspace/bundle/*.whl \
    && chmod +x /workspace/bundle/onstart.sh /workspace/bundle/run_shard.sh \
    && mkdir -p /workspace/data /workspace/results /workspace/status /workspace/control

# The controller passes onstart.sh's contents via --onstart-cmd at create time;
# this default keeps a manually-started container consistent with that path.
CMD ["bash", "/workspace/bundle/onstart.sh"]
