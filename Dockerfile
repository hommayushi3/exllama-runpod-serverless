FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir /app
WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install huggingface_hub runpod
RUN git clone https://github.com/turboderp/exllama
RUN pip install -r exllama/requirements.txt

COPY handler.py /app/handler.py

ENV PYTHONPATH=/app/exllama
ENV MODEL_REPO=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD ["python", "-u", "/app/handler.py"]
