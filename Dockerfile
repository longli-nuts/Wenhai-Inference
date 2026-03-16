FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

USER root

# Install gfortran for aerobulk Fortran compilation and wget for manual install.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran libgfortran5 wget \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

# Install aerobulk manually: not on PyPI, broken build with modern setuptools.
USER root
RUN wget -q https://files.pythonhosted.org/packages/3f/8c/7f68eb7ce0508775ffc33bbc3086c652abe4904fb2b5ecf18c200f4cf752/aerobulk-python-0.4.2.tar.gz -P /tmp && \
    cd /tmp && tar xzf aerobulk-python-0.4.2.tar.gz && \
    cd aerobulk-python-0.4.2 && touch README.md && \
    pip install . --no-build-isolation && \
    cd / && rm -rf /tmp/aerobulk-python-0.4.2*
USER $MAMBA_USER

WORKDIR /app
COPY ./run_wenhai_inference.py .
COPY ./wenhai_inference.py .
COPY ./fetch_copernicus_marine.py .
COPY ./fetch_era5.py .
COPY ./model_manager.py .
COPY ./s3_upload.py .
COPY ./generate_thumbnails.py .

CMD ["python", "run_wenhai_inference.py"]