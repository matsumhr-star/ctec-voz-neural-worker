FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# pkuseg 0.0.25 importa NumPy durante a montagem da wheel, mas não declara
# NumPy corretamente no ambiente isolado de build. Por isso instalamos a base
# primeiro e montamos o pkuseg sem isolamento antes das demais dependências.
RUN python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1" \
    && python -m pip install \
        "numpy==1.25.2" \
        "Cython<3" \
    && python -m pip install --no-build-isolation \
        "pkuseg==0.0.25" \
    && python -m pip install -r requirements.txt

COPY handler.py .
CMD ["python", "-u", "handler.py"]

