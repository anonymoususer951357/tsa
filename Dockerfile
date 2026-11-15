FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/root/miniconda3 \
    VENV_DIR=/opt/tsa-venv \
    TSA_ROOT=/workspace/tsa \
    LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && if ${CONDA_DIR}/bin/conda tos accept --help >/dev/null 2>&1; then \
         ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main ; \
         ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r ; \
       fi \
    && ${CONDA_DIR}/bin/conda create -y -n py311 python=3.11.15 \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/envs/py311/bin:${CONDA_DIR}/bin:${PATH}

RUN python -m venv ${VENV_DIR}
ENV PATH=${VENV_DIR}/bin:${PATH} \
    VIRTUAL_ENV=${VENV_DIR}

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -U pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 torchvision==0.28.0 \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m pip install pillow scikit-image \
    && rm -f /tmp/requirements.txt \
    && python - <<'PY'
import torch, torchvision, numpy, cv2, lmdb, einops, timm, lpips, PIL, skimage
print("torch", torch.__version__, "cuda", torch.version.cuda, "numpy", numpy.__version__)
assert torch.__version__.startswith("2.13.0")
assert torchvision.__version__.startswith("0.28.0")
PY

COPY . ${TSA_ROOT}
WORKDIR ${TSA_ROOT}
RUN chmod +x docker/entrypoint.sh scripts/*.sh \
    && ln -sf ${TSA_ROOT}/docker/entrypoint.sh /usr/local/bin/tsa-entrypoint

ENV DATA_DIR=${TSA_ROOT}/reproduce/data \
    WEIGHTS_DIR=${TSA_ROOT}/reproduce/weights \
    SAVE_DIR=${TSA_ROOT}/save

ENTRYPOINT ["tsa-entrypoint"]
CMD ["bash"]
