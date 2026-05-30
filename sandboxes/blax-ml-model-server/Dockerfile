FROM python:3.11-slim

COPY --from=ghcr.io/blaxel-ai/sandbox:latest /sandbox-api /usr/local/bin/sandbox-api

WORKDIR /blaxel

ENV HOME=/blaxel

COPY sandboxes/blax-ml-model-server/requirements.txt /blaxel/requirements.txt

RUN pip install --no-cache-dir -r /blaxel/requirements.txt

COPY sandboxes/blax-ml-model-server/model_server.py /blaxel/model_server.py

COPY --chmod=755 sandboxes/blax-ml-model-server/entrypoint.sh /entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]