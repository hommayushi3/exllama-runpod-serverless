FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install safetensors==0.3.1 sentencepiece huggingface_hub runpod ninja==1.11.1 pydantic-settings --no-cache-dir
RUN git clone https://github.com/turboderp/exllama.git
RUN pip install -r exllama/requirements.txt --no-cache-dir

COPY handler.py /data/handler.py
COPY settings.py /data/settings.py
COPY __init.py__ /data/__init__.py

ENV PYTHONPATH=/data/exllama
ENV MODEL_REPO=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD [ "python", "-m", "handler" ]
