FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /sec

COPY . .

RUN pip install .

ENV SPELLING_CORRECTION_DOWNLOAD_DIR=/sec/download
ENV SPELLING_CORRECTION_CACHE_DIR=/sec/cache
ENV PYTHONWARNINGS="ignore"

ENTRYPOINT ["/opt/conda/bin/sec"]
