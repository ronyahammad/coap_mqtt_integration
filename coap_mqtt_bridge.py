
import os
import json
import time
import asyncio
import logging
from typing import Dict, Any

import aiocoap
import paho.mqtt.client as mqtt

# -----------------------------
# Environment / Defaults
# -----------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
COAP_URL = os.getenv("COAP_URL", "coap://coap-server:5683/sensor")

# Fallback interval if payload doesn't contain interval_s yet
DEFAULT_INTERVAL = int(os.getenv("DEFAULT_INTERVAL", "5"))

# Topic root; final publish topic becomes: UALG2025_THESIS_IOT_SENSORS/<run_mode>
TOPIC_ROOT = os.getenv("TOPIC_ROOT", "UALG2025_THESIS_IOT_SENSORS")


# -----------------------------
# MQTT helpers
# -----------------------------
def mqtt_connect() -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def _on_disconnect(c, _u, rc, _p=None):
        if rc != 0:
            logging.warning(
                "MQTT disconnected unexpectedly (rc=%s). Will auto-reconnect.", rc)

    client.on_disconnect = _on_disconnect

    backoff = 1
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_start()
            logging.info("Connected to MQTT at %s:%d", MQTT_BROKER, MQTT_PORT)
            return client
        except Exception as e:
            logging.warning(
                "MQTT connect failed: %s. Retrying in %ss...", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


def publish_json(client: mqtt.Client, topic: str, payload: Dict[str, Any]):
    # add timestamp (ns) and ensure run_mode exists
    payload.setdefault("timestamp", int(time.time() * 1e9))
    payload.setdefault("run_mode", "no_rl")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    client.publish(topic, data)


# -----------------------------
# CoAP helpers
# -----------------------------
async def fetch_coap_json(ctx: aiocoap.Context, url: str) -> Dict[str, Any]:
    req = aiocoap.Message(code=aiocoap.GET, uri=url)
    resp = await ctx.request(req).response
    try:
        # payload is already JSON from the coap_server
        data = json.loads(resp.payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("CoAP returned non-dict JSON")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to decode CoAP JSON: {e}") from e


# -----------------------------
# Main async loop
# -----------------------------
async def bridge_loop():
    logging.info("Starting CoAP→MQTT bridge")
    mqttc = mqtt_connect()
    ctx = await aiocoap.Context.create_client_context()

    # pacing
    interval_s = DEFAULT_INTERVAL
    last_warn_missing_runmode = 0

    while True:
        try:
            # 1) Pull from CoAP
            data = await fetch_coap_json(ctx, COAP_URL)

            # 2) Determine run_mode + publish topic
            run_mode = data.get("run_mode", "no_rl")
            if not isinstance(run_mode, str):
                run_mode = "no_rl"
            topic = f"{TOPIC_ROOT}/{run_mode}"

            # 3) Publish verbatim (plus timestamp default in publish_json)
            publish_json(mqttc, topic, data)

            # 4) Update sleep interval from payload (agent-controlled)
            new_interval = data.get("interval_s", interval_s)
            try:
                new_interval = int(new_interval)
                if new_interval not in (5, 30, 60):
                    # guardrail: keep within expected set
                    new_interval = interval_s
            except Exception:
                # leave as-is if not parseable
                new_interval = interval_s

            interval_s = new_interval

            # Optional: warn if run_mode missing (not every loop to avoid spam)
            if "run_mode" not in data and (time.time() - last_warn_missing_runmode) > 60:
                logging.warning("Payload missing 'run_mode'; defaulting to no_rl. "
                                "Set RUN_MODE in coap_server to 'no_rl' or 'ddqn'.")
                last_warn_missing_runmode = time.time()

        except aiocoap.error.RequestTimedOut:
            logging.warning("CoAP request timed out; will retry.")
        except Exception as e:
            logging.error("Bridge loop error: %s", e, exc_info=True)

        # 5) Sleep according to current interval (agent sets this through control actions)
        await asyncio.sleep(max(1, int(interval_s)))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    try:
        asyncio.run(bridge_loop())
    except KeyboardInterrupt:
        logging.info("Bridge interrupted by user. Exiting.")


if __name__ == "__main__":
    main()

