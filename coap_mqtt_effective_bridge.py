import os
import json
import time
import asyncio
import logging
from collections import deque
from statistics import median, mean
from typing import Any, Dict, Optional, Tuple

import aiocoap
import paho.mqtt.client as mqtt

# -----------------------------
# Environment / Defaults
# -----------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC_EFFECTIVE",
                       "UALG2025_THESIS_IOT_SENSORS/ddqn_effective")

COAP_URL = os.getenv("COAP_URL", "coap://coap-server/sensor")

# Rolling window for summarization and potential batch flush
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "30"))
# how many items to dump when bufT is high
MAX_BATCH = int(os.getenv("MAX_BATCH", "10"))

# When we only have numeric deltas (no binned dT), use these absolute thresholds (°C) to grade delta
DELTA_THRESHOLDS = tuple(
    float(x) for x in os.getenv("DELTA_THRESHOLDS", "0.15,0.75").split(",")
)
# (low, high) means: <low -> dT=0 (small), <high -> dT=1 (medium), else dT=2 (large)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# -----------------------------
# Helpers
# -----------------------------


def safe_int(d: Dict[str, Any], key: str, default: int = 0, lo: int = None, hi: int = None) -> int:
    try:
        v = int(d.get(key, default))
        if lo is not None and v < lo:
            v = lo
        if hi is not None and v > hi:
            v = hi
        return v
    except Exception:
        return default


def compute_dt_bin(raw: Dict[str, Any]) -> int:
    """
    Prefer the already-binned 'dT' from the sensor if present.
    Fallback to computing bins from an absolute delta: |temperature - ema|.
    """
    if "dT" in raw:
        return safe_int(raw, "dT", 0, 0, 2)
    # fallback: try to compute from raw fields
    try:
        t = float(raw.get("temperature"))
        ema = float(raw.get("ema", t))
        delta = abs(t - ema)
        low, high = DELTA_THRESHOLDS
        if delta < low:
            return 0
        if delta < high:
            return 1
        return 2
    except Exception:
        return 0


def needs_flush(bufT: int, critical: int, fidelity: int) -> bool:
    return bufT >= 2 or (critical == 1 and fidelity == 2)


def summarize_temperature(series: deque) -> Tuple[Optional[float], Optional[float], int]:
    if not series:
        return None, None, 0
    vals = list(series)
    return median(vals), mean(vals), len(vals)

# -----------------------------
# MQTT client
# -----------------------------


def make_mqtt() -> mqtt.Client:
    client = mqtt.Client(clean_session=True)
    client.enable_logger()
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_start()
    return client

# -----------------------------
# CoAP fetch
# -----------------------------


async def fetch_sensor(ctx, url: str) -> Optional[Dict[str, Any]]:
    req = aiocoap.Message(code=aiocoap.GET, uri=url)
    try:
        resp = await ctx.request(req).response
        data = json.loads(resp.payload.decode("utf-8"))
        return data
    except Exception as e:
        logging.warning("CoAP fetch error: %s", e)
        return None

# -----------------------------
# Main bridge loop
# -----------------------------


async def bridge_loop():
    logging.info("Starting ddqn_effective bridge: %s -> %s",
                 COAP_URL, MQTT_TOPIC)

    # Rolling buffers for local summarization & batch (temperature + optional humidity)
    temp_buf = deque(maxlen=ROLLING_WINDOW)
    hum_buf = deque(maxlen=ROLLING_WINDOW)

    # CoAP client context and MQTT
    ctx = await aiocoap.Context.create_client_context()
    mqttc = make_mqtt()

    interval_s = 5
    last_warn_missing_runmode = 0.0

    while True:
        start_ts = time.time()

        raw = await fetch_sensor(ctx, COAP_URL)
        if raw is None:
            await asyncio.sleep(max(1, interval_s))
            continue

        # --- update local windows
        t = raw.get("temperature")
        if isinstance(t, (int, float)) or (isinstance(t, str) and t.replace('.', '', 1).isdigit()):
            try:
                temp_buf.append(float(t))
            except Exception:
                pass

        h = raw.get("humidity")
        if isinstance(h, (int, float)) or (isinstance(h, str) and h.replace('.', '', 1).isdigit()):
            try:
                hum_buf.append(float(h))
            except Exception:
                pass

        # --- read current action snapshot from the payload
        qos = safe_int(raw, "qos", 0, 0, 2)
        fidelity = safe_int(raw, "fidelity", 0, 0, 2)
        critical = safe_int(raw, "critical", 0, 0, 1)
        bufT = safe_int(raw, "bufT", 0, 0, 2)

        # sensor may send 'interval_s' or 'interval'
        interval_s = safe_int(raw, "interval_s", safe_int(
            raw, "interval", interval_s))
        if interval_s not in (5, 30, 60):
            # keep sane
            interval_s = min(60, max(1, interval_s))

        # --- compute/confirm dT bin
        dT = compute_dt_bin(raw)

        # --- build effective payload starting from raw
        eff = dict(raw)  # shallow copy
        eff["timestamp_bridge"] = time.time()
        eff["stage"] = "effective"
        # so the Influx bridge can route/measure
        eff["run_mode"] = "ddqn_effective"
        eff["used_qos"] = qos

        # --- fidelity-based summarization when quiet AND low fidelity
        # Condition: small delta -> summarize to reduce network cost
        did_summarize = False
        if fidelity == 0 and dT == 0 and len(temp_buf) >= 3:
            med, avg, n = summarize_temperature(temp_buf)
            if med is not None:
                eff["temperature_eff"] = med
                eff["temperature_eff_mean"] = avg
                eff["summary_kind"] = "median"
                eff["summary_window"] = n
                # Option: replace raw 'temperature' with the summary to model payload reduction
                eff["temperature"] = med
                did_summarize = True

        # --- backlog flush when bufT is high or we are in burst mode (critical & high fidelity)
        batch = []
        if needs_flush(bufT, critical, fidelity):
            # Take up to MAX_BATCH recent local samples as a proxy for "unsent" data
            # (For full fidelity, teach the sensor to return true history.)
            # We include only temperature here; add humidity if you want.
            vals = list(temp_buf)
            take = min(MAX_BATCH, len(vals))
            if take > 0:
                # oldest to newest of the last 'take' values
                batch = vals[-take:]
            eff["history_temperature"] = batch
            eff["batch_len"] = len(batch)

        # --- publish effective payload with the agent-selected QoS
        payload = json.dumps(eff, separators=(",", ":"))
        try:
            mqttc.publish(MQTT_TOPIC, payload, qos=qos, retain=False)
        except Exception as e:
            logging.error("MQTT publish error: %s", e)

        # --- pacing: honor the current interval
        # If we flushed a batch, we still keep the regular pacing; adjust if you want burst pulses.
        elapsed = time.time() - start_ts
        sleep_s = max(0.5, interval_s - elapsed)
        await asyncio.sleep(sleep_s)


def main():
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")
    try:
        asyncio.run(bridge_loop())
    except KeyboardInterrupt:
        logging.info("Effective bridge interrupted by user. Exiting.")


if __name__ == "__main__":
    main()
