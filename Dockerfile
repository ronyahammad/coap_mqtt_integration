FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install aiocoap paho-mqtt

CMD ["python", "coap_mqtt_bridge.py"]
