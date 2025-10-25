FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y netcat-openbsd && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 && \
    python -m pip install aiocoap paho-mqtt influxdb-client numpy python-dotenv

CMD ["python", "coap_mqtt_bridge.py"]