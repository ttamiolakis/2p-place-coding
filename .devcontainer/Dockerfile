FROM mcr.microsoft.com/devcontainers/miniconda:0-3
RUN sudo apt-get update
RUN sudo apt-get install -y xorg

ADD environment.yml /tmp/environment.yml
RUN conda update -n base -c conda-forge conda
RUN conda env create -n placecoding -f /tmp/environment.yml

RUN echo "conda activate placecoding" >> ~/.bashrc
ENV PATH /opt/conda/envs/placecoding/bin:$PATH

ENV CONDA_DEFAULT_ENV placecoding