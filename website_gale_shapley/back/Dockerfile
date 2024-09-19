FROM python:3.9.6-slim

ENV SERVICE_FOLDER_NAME=back
WORKDIR /${SERVICE_FOLDER_NAME}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

RUN set -ex \
    && RUN_DEPS=" \
    libpq-dev \
    git \
    " \
    && seq 1 8 | xargs -I{} mkdir -p /usr/share/man/man{} \
    && apt-get update && apt-get install -y --no-install-recommends $RUN_DEPS \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /${SERVICE_FOLDER_NAME}/

EXPOSE 8000

RUN chmod +x /${SERVICE_FOLDER_NAME}/entrypoint.sh

ENTRYPOINT /$SERVICE_FOLDER_NAME/entrypoint.sh
