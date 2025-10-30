FROM ubuntu:22.04

WORKDIR /app/
COPY requirements.txt requirements.txt
COPY SketchGraphs/ SketchGraphs/
COPY occenv.yml occenv.yml

ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages and Python
RUN apt-get update -y \
    && apt-get install -y curl wget software-properties-common build-essential git ca-certificates gnupg libgl1 fuse zsh

# Install Python 3.10 (available natively in Ubuntu 22.04)
RUN apt-get update -y \
    && apt-get install -y python3.10 python3.10-dev python3.10-venv \
    && ls -la /usr/bin/python3.10 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && python -m pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- Xvfb & fonts --------------------------------------------------
    xvfb \
    fonts-dejavu \
    fonts-liberation \
    # --- Qt / X11 run‑time libs ---------------------------------------
    # provides the virtual libgl1
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-util1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxkbcommon-x11-0 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN python -m pip install -r requirements.txt
RUN python -m pip install -e ./SketchGraphs

# Fix GPG issues and install FreeCAD
RUN apt-get update --allow-insecure-repositories \
    && apt-get install -y --allow-unauthenticated gnupg2 \
    && add-apt-repository ppa:freecad-maintainers/freecad-stable -y \
    && apt-get update -y \
    && apt-get install -y freecad

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

# Add Conda to PATH (append so /usr/bin/python remains default)
ENV PATH=$PATH:$CONDA_DIR/bin

# Accept TOS for required conda channels (no conda init needed for conda run)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create occenv conda environment from yml file
RUN conda env create -f occenv.yml && \
    conda clean -afy


CMD ["bash"]