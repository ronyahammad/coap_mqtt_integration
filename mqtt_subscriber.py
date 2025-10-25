import logging
import paho.mqtt.client as mqtt
import time


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        logging.info("Connected to MQTT Broker!")
    else:
        logging.error(f"Failed to connect, return code {rc}")


def on_message(client, userdata, message):
    # Skip control channel spam; print everything else (no_rl, ddqn, summaries)
    if message.topic.endswith("/control"):
        return
    try:
        payload = message.payload.decode("utf-8")
    except Exception:
        payload = str(message.payload)
    logging.info(f"[{message.topic}] {payload}")


mqttBroker = "mosquitto"
mqtt_topic = "UALG2025_THESIS_IOT_SENSORS/#"

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_message = on_message


def connect_mqtt():
    while True:
        try:
            mqttc.connect(mqttBroker)
            mqttc.loop_start()
            logging.info("Connected to MQTT Broker!")
            return True
        except ConnectionRefusedError:
            logging.warning(
                "MQTT broker not available yet, retrying in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    connect_mqtt()
    mqttc.subscribe(mqtt_topic)
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        mqttc.loop_stop()
        mqttc.disconnect()
