FROM ubuntu:20.04

WORKDIR /app/
COPY requirements.txt requirements.txt
COPY SketchGraphs/ SketchGraphs/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y curl wget software-properties-common build-essential git \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update -y \
    && apt-get install -y python3.10 python3.10-distutils libgl1 fuse zsh \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt
RUN python -m pip install -e ./SketchGraphs

RUN add-apt-repository ppa:freecad-maintainers/freecad-stable -y \
    && apt-get update -y \
    && apt-get install -y freecad

CMD ["bash"]