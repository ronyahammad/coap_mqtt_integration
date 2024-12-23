import logging
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    logging.info(f"Received: {message.payload.decode('utf-8')}")


mqttBroker = "mqtt.eclipseprojects.io"
mqtt_topic = "SENSOR/DATA"

mqttc = mqtt.Client()
mqttc.on_message = on_message

mqttc.connect(mqttBroker)
mqttc.subscribe(mqtt_topic)

logging.basicConfig(level=logging.INFO)

try:
    mqttc.loop_forever()
except KeyboardInterrupt:
    print("Exiting program...")
finally:
    mqttc.disconnect()
