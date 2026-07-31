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

# O pkuseg 0.0.25 foi publicado antes do Python 3.11 e referencia
# longintrepr.h pelo caminho antigo. Baixamos o código-fonte, ajustamos
# o include para Python 3.11 e instalamos o pacote localmente.
RUN python -m pip install --upgrade \
        "pip==25.1.1" \
        "setuptools==75.8.0" \
        "wheel==0.45.1" \
    && python -m pip install \
        "numpy==1.25.2" \
        "Cython<3" \
    && mkdir -p /tmp/pkuseg-src \
    && python -m pip download --no-deps --no-binary=:all: \
        "pkuseg==0.0.25" -d /tmp/pkuseg-src \
    && tar -xzf /tmp/pkuseg-src/pkuseg-0.0.25.tar.gz -C /tmp/pkuseg-src \
    && find /tmp/pkuseg-src/pkuseg-0.0.25 -type f \
        \( -name '*.c' -o -name '*.cpp' -o -name '*.h' \) \
        -exec sed -i 's/#include "longintrepr.h"/#include "cpython\/longintrepr.h"/g' {} + \
    && python -m pip install --no-build-isolation /tmp/pkuseg-src/pkuseg-0.0.25 \
    && python -m pip install -r requirements.txt \
    && rm -rf /tmp/pkuseg-src

COPY handler.py .
CMD ["python", "-u", "handler.py"]
