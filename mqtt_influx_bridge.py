
import os
from dotenv import load_dotenv, find_dotenv
# Load .env without overriding real env (Compose)
load_dotenv(find_dotenv(usecwd=True), override=False)


def getenv_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"{name} is empty. Set it via docker-compose or .env")
    return v

import json
import time
import logging
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# -----------------------------
# Environment
# -----------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "UALG2025_THESIS_IOT_SENSORS/#")

INFLUX_URL = os.getenv("INFLUX_URL", "http://influxdb:8086")
INFLUX_TOKEN = getenv_required("INFLUX_TOKEN")
INFLUX_ORG = getenv_required("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "iot_data")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------------
# Helpers
# -----------------------------
def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _now_ns() -> int:
    return int(time.time() * 1e9)


def _run_mode_from_topic_or_payload(topic: str, data: Dict[str, Any]) -> str:
    # e.g., UALG2025_THESIS_IOT_SENSORS/no_rl  or  .../ddqn
    last = topic.split("/")[-1].lower()
    if last in ("no_rl", "ddqn", "ddqn_effective"):
        return last
    # fallback to payload
    v = str(data.get("run_mode", "no_rl")).lower()
    if v == "ddqn_effective":
        return "ddqn_effective"
    if v == "ddqn": return "ddqn"
    return "no_rl"


# -----------------------------
# Influx writer
# -----------------------------
class InfluxWriter:
    def __init__(self):
        self.client = InfluxDBClient(
            url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=30000)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def write_point(self, p: Point):
        self.write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)


# -----------------------------
# Builders
# -----------------------------
def build_summary_point(data: Dict[str, Any]) -> Point:
    """
    Map ddqn_episode_summary into fields.
    """
    p = Point("ddqn_episode_summary").tag("run_mode", "ddqn")
    for k in ("episode", "total_reward", "total_energy_mAh", "avg_latency_ms",
              "on_time_rate", "drop_rate", "late_rate", "steps"):
        if k in data:
            vnum = _as_float(data[k])
            if vnum is None:
                vnum = _as_int(data[k])
            if isinstance(vnum, (float, int)):
                p = p.field(k, vnum)
    ts = data.get("timestamp")
    if isinstance(ts, int):
        p = p.time(ts, WritePrecision.NS)
    else:
        p = p.time(_now_ns(), WritePrecision.NS)
    return p


def build_control_point(data: Dict[str, Any]) -> Point:
    p = Point("control_actions")
    # normalize interval -> interval_s
    interval = data.get("interval_s", data.get("interval"))
    if interval is not None:
        ival = _as_int(interval)
        if ival is not None:
            p = p.field("interval_s", ival)
    for k in ("qos", "fidelity", "critical"):
        v = _as_int(data.get(k))
        if v is not None:
            p = p.field(k, v)
    # NEW: p_crit is optional but great for panels
    pcrit = _as_float(data.get("p_crit"))
    if pcrit is not None:
        p = p.field("p_crit", pcrit)

    ts = data.get("timestamp")
    p = p.time(ts if isinstance(ts, int) else _now_ns(), WritePrecision.NS)
    return p



def build_telemetry_point(data: Dict[str, Any], topic: str) -> Point:
    """
    Map regular telemetry into sensor_data (no_rl) or ddqn_data (ddqn).
    """
    run_mode = _run_mode_from_topic_or_payload(topic, data)
    measurement = {"no_rl": "sensor_data", "ddqn": "ddqn_data",
                   "ddqn_effective": "ddqn_effective"}[run_mode]
    p = Point(measurement).tag("run_mode", run_mode)

    # Tag packet status if present
    pkt_status = data.get("packet_status")
    if isinstance(pkt_status, str):
        p = p.tag("packet_status", pkt_status)
    
    v = _as_float(data.get("latency_ms"))
    if v is not None:
        p = p.field("latency_ms", v)

    v = _as_int(data.get("true_critical"))
    if v is not None:
        p = p.field("true_critical", v)

    # Temperature
    temp = _as_float(data.get("temperature"))
    if temp is not None:
        p = p.field("temperature", temp)

    # Battery & energy
    for k in ("battery_mAh", "battery_pct", "energy_mAh_used"):
        v = _as_float(data.get(k))
        if v is not None:
            p = p.field(k, v)

    # EMA / deltaT (continuous)
    for k in ("ema", "deltaT"):
        v = _as_float(data.get(k))
        if v is not None:
            p = p.field(k, v)

    # Binned/discrete state variables
    for k in ("dT", "bufT", "B", "L", "D", "anomT", "I_cur", "Q_cur", "F_cur", "true_critical"):
        v = _as_int(data.get(k))
        if v is not None:
            p = p.field(k, v)

    # Action snapshot
    q = _as_int(data.get("qos"))
    if q is not None:
        p = p.field("qos", q)
    ival = _as_int(data.get("interval_s"))
    if ival is not None:
        p = p.field("interval_s", ival)
    f = _as_int(data.get("fidelity"))
    if f is not None:
        p = p.field("fidelity", f)
    c = _as_int(data.get("critical"))
    if c is not None:
        p = p.field("critical", c)

    # Latency (if DDQN pipeline emits it)
    v = _as_float(data.get("latency_ms"))
    if v is not None:
        p = p.field("latency_ms", v)

        # --- Effective-only extras (for ddqn_effective dashboards)
    if run_mode == "ddqn_effective":
        v = _as_int(data.get("used_qos"))
        if v is not None:
            p = p.field("used_qos", v)
        v = _as_float(data.get("temperature_eff"))
        if v is not None:
            p = p.field("temperature_eff", v)
        v = _as_float(data.get("temperature_eff_mean"))
        if v is not None:
            p = p.field("temperature_eff_mean", v)
        v = _as_int(data.get("batch_len"))
        if v is not None:
            p = p.field("batch_len", v)
        sk = data.get("summary_kind")
        if isinstance(sk, str):
            p = p.tag("summary_kind", sk)
        v = _as_int(data.get("summary_window"))
        if v is not None:
            p = p.field("summary_window", v)


    # Timestamp
    ts = data.get("timestamp")
    if isinstance(ts, int):
        p = p.time(ts, WritePrecision.NS)
    else:
        p = p.time(_now_ns(), WritePrecision.NS)

    return p


# -----------------------------
# MQTT Bridge
# -----------------------------
class Bridge:
    def __init__(self):
        self.influx = InfluxWriter()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logging.info("MQTT connected. Subscribing to %s", MQTT_TOPIC)
            client.subscribe(MQTT_TOPIC, qos=1)
        else:
            logging.error("MQTT connect failed with rc=%s", rc)

    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if not isinstance(data, dict):
                logging.warning("Ignoring non-dict JSON on %s", msg.topic)
                return

            # (1) Episode summaries
            if msg.topic.endswith("/ddqn_episode_summary"):
                point = build_summary_point(data)
                self.influx.write_point(point)
                return

            # (2) Control actions
            if msg.topic.endswith("/control"):
                point = build_control_point(data)
                self.influx.write_point(point)
                return

            # (3) Regular telemetry → sensor_data (no_rl) or ddqn_data (ddqn)
            point = build_telemetry_point(data, msg.topic)
            self.influx.write_point(point)

        except json.JSONDecodeError:
            logging.error("Invalid JSON on topic %s", msg.topic)
        except Exception as e:
            logging.error("on_message error: %s", e, exc_info=True)

    def on_disconnect(self, client, userdata, rc, properties=None):
        if rc != 0:
            logging.warning(
                "MQTT disconnected unexpectedly (rc=%s). Reconnecting...", rc)

    def run(self):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.on_disconnect = self.on_disconnect

        backoff = 1
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever(retry_first_connection=True)
            except Exception as e:
                logging.warning(
                    "MQTT connection error: %s; retrying in %ss", e, backoff)
                time.sleep(backoff)
                backoff = min(30, backoff * 2)


def main():
    logging.info("Starting MQTT→Influx bridge (bucket=%s)", INFLUX_BUCKET)
    Bridge().run()


if __name__ == "__main__":
    main()
