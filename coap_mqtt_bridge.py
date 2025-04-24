import asyncio
import json
import logging
import paho.mqtt.client as mqtt
from aiocoap import *

mqttBroker = "mqtt.eclipseprojects.io"
mqtt_topic = "SENSOR/DATA"

mqttc = mqtt.Client()
mqttc.connect(mqttBroker)

# Create global protocol context
protocol = None


async def fetch_coap_data(protocol):
    request = Message(code=GET, uri="coap://coap-server:5683/sensor")

    try:
        response = await protocol.request(request).response
        sensor_data = response.payload.decode('utf-8')
        mqttc.publish(mqtt_topic, sensor_data)
        print(f"Published to MQTT: {sensor_data}")
    except Exception as e:
        logging.error(f"CoAP Request Failed: {e}")


async def main():
    global protocol
    logging.info("Starting CoAP-MQTT Bridge...")

    # Create CoAP context once and reuse
    protocol = await Context.create_client_context()

    while True:
        await fetch_coap_data(protocol)
        await asyncio.sleep(5)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mqttc.loop_start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bridge interrupted. Exiting...")
    finally:
        mqttc.loop_stop()
        mqttc.disconnect()
