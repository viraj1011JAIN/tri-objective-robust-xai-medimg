# Base: PyTorch 2.9.0 + CUDA 13.0 + cuDNN 9 runtime
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

# Workdir inside container
WORKDIR /workspace

# Basic OS deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        wget \
        ca-certificates \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy dependency manifests first (Docker layer caching)
COPY requirements.txt pyproject.toml environment.yml ./ 

# Install Python deps (image already has Python + pip)
RUN pip install --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

ENV PYTHONPATH=/workspace

# Default shell – you'll usually override this with 'docker run -it ...'
CMD ["bash"]
