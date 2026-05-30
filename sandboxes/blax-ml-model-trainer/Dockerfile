FROM python:3.11-slim

COPY --from=ghcr.io/blaxel-ai/sandbox:latest /sandbox-api /usr/local/bin/sandbox-api

WORKDIR /blaxel

ENV HOME=/blaxel

COPY sandboxes/blax-ml-model-trainer/requirements.txt /blaxel/requirements.txt

RUN pip install --no-cache-dir -r /blaxel/requirements.txt

COPY sandboxes/blax-ml-model-trainer/model_trainer.py /blaxel/model_trainer.py

COPY sandboxes/blax-ml-model-trainer/csv_profiler.py /blaxel/csv_profiler.py

ENTRYPOINT ["/usr/local/bin/sandbox-api"]