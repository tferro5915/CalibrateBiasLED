FROM mcr.microsoft.com/devcontainers/miniconda:0-3

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN conda install --yes --freeze-installed -n base conda-libmamba-solver  && \
	conda config --set solver libmamba

COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update \
        -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp && \
    conda clean -afy
	

