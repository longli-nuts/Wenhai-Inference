FROM mambaorg/micromamba:1.5.6-focal-cuda-11.8.0

USER root

# Install gfortran for aerobulk Fortran compilation and wget for manual install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran libgfortran5 wget \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

ENV CONDA_OVERRIDE_CUDA=11.8

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yaml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes


ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# Install aerobulk manually: not on PyPI, broken build with modern setuptools.
USER root
RUN wget -q https://files.pythonhosted.org/packages/3f/8c/7f68eb7ce0508775ffc33bbc3086c652abe4904fb2b5ecf18c200f4cf752/aerobulk-python-0.4.2.tar.gz -P /tmp && \
    cd /tmp && tar xzf aerobulk-python-0.4.2.tar.gz && \
    cd aerobulk-python-0.4.2 && touch README.md && \
    micromamba run -n base python -m pip install "setuptools<60" && \
    micromamba run -n base python -m pip install . --no-build-isolation && \
    cd / && rm -rf /tmp/aerobulk-python-0.4.2*
USER $MAMBA_USER

WORKDIR /app
COPY ./run_wenhai_inference.py .
COPY ./wenhai_inference.py .
COPY ./fetch_copernicus_marine.py .
COPY ./fetch_ifs.py .
COPY ./model_manager.py .
COPY ./s3_upload.py .
COPY ./generate_thumbnails.py .
COPY ./add_metadata.py .

CMD ["python", "run_wenhai_inference.py"]
