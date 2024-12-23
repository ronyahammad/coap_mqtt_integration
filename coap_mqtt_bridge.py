import asyncio
import json
import logging
import paho.mqtt.client as mqtt
from aiocoap import *

mqttBroker = "mqtt.eclipseprojects.io"
mqtt_topic = "SENSOR/DATA"

mqttc = mqtt.Client()
mqttc.connect(mqttBroker)


async def fetch_coap_data():

    protocol = await Context.create_client_context()
    request = Message(code=GET, uri="coap://127.0.0.1:5683/sensor")

    try:
        response = await protocol.request(request).response
        sensor_data = response.payload.decode('utf-8')
        mqttc.publish(mqtt_topic, sensor_data)
        print(f"Published to MQTT: {sensor_data}")
    except Exception as e:
        print(f"Failed to fetch or publish data: {e}")


async def main():
    while True:
        await fetch_coap_data()
        await asyncio.sleep(5)  

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mqttc.loop_start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        mqttc.loop_stop()
        mqttc.disconnect()
