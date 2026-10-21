# NVIDIA CUDA 12.8 runtime on Ubuntu 24.04 (distro Python 3.12).
#
# Why 24.04 and not 22.04: soup-cli declares requires-python ">=3.11,<3.14" and Ubuntu
# 22.04's python3 is 3.10. pip does not fail on that. It quietly resolves the newest
# soup-cli whose requires-python still admits 3.10, so an image tagged 0.77.0 would hold
# 0.76.x. tests/test_dockerfile_python.py derives the allowed range from pyproject.toml.
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

# Set environment variables. The venv is first on PATH: Ubuntu 24.04 marks the system
# Python as externally managed (PEP 668), so `pip install` into it is refused.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Python 3.12 is Ubuntu 24.04's native python3. No deadsnakes PPA: it has had repeated
# connection timeouts from GHA runners.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create the venv (it provides `python`, `pip` and `soup` on PATH), then refuse to go on
# with a Python outside soup-cli's requires-python, before any wheel is downloaded.
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), 'soup-cli needs Python 3.11-3.13, this image has ' + sys.version"

# Set working directory
WORKDIR /workspace

# The soup-cli release to install from PyPI, never the local source. docker.yml passes the
# version of the git tag it was started for; a local build must pass one too:
#   docker build --build-arg SOUP_VERSION=0.77.0 .
# There is deliberately no default: an unpinned install is how a tag and the package inside
# the image drift apart.
ARG SOUP_VERSION
RUN printf '%s' "${SOUP_VERSION}" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' \
    || { echo "error: pass the soup-cli release to install, e.g. --build-arg SOUP_VERSION=0.77.0" >&2; exit 1; }

# v0.71.0 split the training stack into the [train] extra: include it so the GPU image can
# still fine-tune.
RUN pip install --no-cache-dir "soup-cli[train,serve,data,eval]==${SOUP_VERSION}"

# Fail the build unless the package that landed is exactly the requested release and its
# console script is on PATH.
RUN python -c "import importlib.metadata as md, os; got = md.version('soup-cli'); want = os.environ['SOUP_VERSION']; assert got == want, f'installed soup-cli {got}, requested {want}'" \
    && command -v soup

# Default entrypoint and command
ENTRYPOINT ["soup"]
CMD ["--help"]
