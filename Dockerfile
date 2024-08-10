FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04
WORKDIR /app

RUN set -xe \
 && apt update \
 && apt install python3 python3-pip

COPY . .
ENTRYPOINT ["sleep", "inf"]
