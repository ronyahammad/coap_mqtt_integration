import asyncio
import logging
import json
import math
import time
import numpy as np
import aiocoap.resource as resource
import aiocoap
import paho.mqtt.client as mqtt
import os
from collections import deque

# ====== Config from environment ======
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
CONTROL_TOPIC = os.getenv(
    "CONTROL_TOPIC", "UALG2025_THESIS_IOT_SENSORS/control")
RUN_MODE = os.getenv("RUN_MODE", "no_rl")
ANOM_THRESHOLD = float(os.getenv("ANOM_THRESHOLD", "2.5"))
A_SCALE = float(os.getenv("A_SCALE", "1.0"))

# ====== Sensor / Environment Model ======
CRIT_THRESH = 4  # keep in sync with simulator/agent



def compute_true_criticality(state_tuple):
    """
    state_tuple = (dT, bufT, B, L, D, I_cur, Q_cur, F_cur, anomT)
    """
    dT, bufT, B, L, D, *_rest, anomT = state_tuple
    if int(anomT) == 1 or int(bufT) == 2 or int(L) == 2 or int(D) == 2:
        return 1
    score = int(dT) + int(bufT) + (2 - int(B)) + int(L) + int(D) + int(anomT)
    return 1 if score >= CRIT_THRESH else 0


def simulate_link_latency_and_status(qos: int, interval_s: int, fidelity: int):
    """
    Returns (latency_ms, packet_status, is_delivered).
    packet_status ∈ {"on_time","late","dropped"}; 'late' if latency >= 300ms.
    Mirrors the simplified model used in your simulator.
    """
    import random
    base_delay = np.random.normal(100 + 50*(2 - qos), 20)  # QoS0 slower on avg
    delay = float(np.clip(base_delay + 1000/interval_s + 12*fidelity, 0, 800))
    loss_chance = 0.18*(2 - qos) + 0.008*delay/100 + 0.012*fidelity
    is_delivered = (random.random() > loss_chance)
    if not is_delivered:
        return 1000.0, "dropped", False
    packet_status = "on_time" if delay < 300.0 else "late"
    return delay, packet_status, True


class RealisticEnvSensor:
    def __init__(self):
        self.t = 0
        # temperature signal
        self.period = 288
        self.ampl = 10.0
        self.bias = 20.0
        self.noise_std = 0.2

        # EMA
        self.temp_ema = None
        self.delta_alpha = 0.1
        self.delta_std = 1.0

        # Battery
        self.battery_capacity = 5.0
        self.battery_mAh = self.battery_capacity

        # Queue / anomaly
        self.backlog_T = 0
        self.capacity = 300
        self.anom_threshold = ANOM_THRESHOLD

        # Sliding windows for L/D bins
        self.delay_window = deque(maxlen=20)
        self.loss_window = deque(maxlen=20)

        # Constants (kept aligned with simulator)
        base_a = [0.2, 0.6, 1.2]
        # accumulation per dT bin
        self.a_values = [a * A_SCALE for a in base_a]
        # drain factor by fidelity
        self.kappa_F = [0.5, 1.0, 2.0]
        self.phi_f = [0.6, 1.0, 1.6]
        self.overhead_q = [1.0, 1.15, 1.35]
        self.I_tx_base = 120.0  # mA
        self.I_base = 5.0       # mA
        self.I_sense = 2.0      # mA
        self.t_tx_base = 0.1    # s
        # (kept here for historical parity; not used directly)
        self.beta_f = 12
        

        # default baseline action (can be overridden via control)
        self.qos = 0
        self.interval = 5
        self.fidelity = 0
        self.critical = 0

    def apply_action(self, qos, interval, fidelity, critical):
        self.qos = int(qos)
        self.interval = int(interval)
        self.fidelity = int(fidelity)
        self.critical = int(critical)

    def temp_wave(self):
        return (self.bias
                + self.ampl * math.sin(2 * math.pi * self.t / self.period)
                + np.random.normal(0, self.noise_std))

    def update_ema(self, x: float):
        self.temp_ema = x if self.temp_ema is None else self.delta_alpha * \
            x + (1 - self.delta_alpha) * self.temp_ema

    @staticmethod
    def bin_value(value, thresholds):
        if value < thresholds[0]:
            return 0
        elif value < thresholds[1]:
            return 1
        return 2

    def update_backlog(self, dT_bin: int, burst: int) -> int:
        # accumulate & drain a single temperature queue
        self.backlog_T = min(
            self.capacity, self.backlog_T + self.a_values[dT_bin])
        drain_cap = self.kappa_F[self.fidelity] * \
            (5 / self.interval) * (1 + burst)
        drain = min(self.backlog_T, drain_cap)
        self.backlog_T -= drain
        ratio_T = self.backlog_T / self.capacity
        return 0 if ratio_T < 1/3 else (1 if ratio_T < 2/3 else 2)

    def update_battery(self, burst: int):
        duty = min(1.0, (self.t_tx_base *
                   self.phi_f[self.fidelity]) / self.interval * (1 + burst))
        I_tx = self.I_tx_base * self.overhead_q[self.qos] * duty
        I_avg = self.I_base + self.I_sense + I_tx
        k = 1.07
        delta_mAh = (I_avg**k) / 3600.0
        self.battery_mAh = max(0.0, self.battery_mAh - delta_mAh)
        battery = self.battery_mAh / self.battery_capacity
        B_bin = 0 if battery < 0.3 else (1 if battery < 0.7 else 2)
        return B_bin, delta_mAh

    def compute_loss_rate(self) -> float:
        if len(self.loss_window) == 0:
            return 0.0
        return 1.0 - (sum(self.loss_window) / len(self.loss_window))

    def compute_avg_delay(self) -> float:
        if len(self.delay_window) == 0:
            return 0.0
        return sum(self.delay_window) / len(self.delay_window)

    def step(self):
        # 1) new reading
        temp = self.temp_wave()
        self.t += 1
        prev_ema = self.temp_ema if self.temp_ema is not None else temp
        self.update_ema(temp)
        deltaT = abs(temp - self.temp_ema) / (self.delta_std + 1e-8)
        dT_bin = self.bin_value(deltaT, [1, 3])

        # 2) burst if crit & high-fidelity (same logic as simulator)
        burst = 1 if (self.critical == 1 and self.fidelity == 2) else 0

        # 3) backlog & battery
        bufT = self.update_backlog(dT_bin, burst)
        B_bin, energy_used = self.update_battery(burst)

        # 4) network model for this packet (latency/status + sliding windows)
        latency_ms, packet_status, is_delivered = simulate_link_latency_and_status(
            self.qos, self.interval, self.fidelity)
        # update windows with this observation
        self.delay_window.append(latency_ms if is_delivered else 1000.0)
        self.loss_window.append(1 if is_delivered else 0)

        # 5) loss/delay bins over sliding window
        loss_rate = self.compute_loss_rate()
        L_bin = self.bin_value(loss_rate, [0.1, 0.3])
        avg_delay = self.compute_avg_delay()
        D_bin = self.bin_value(avg_delay, [100, 500])

        # 6) anomaly flag on temperature
        anomT = 1 if deltaT > self.anom_threshold else 0

        # 7) reflect chosen controls into discretized state
        # default to 30s bin if unexpected
        I_cur = {5: 0, 30: 1, 60: 2}.get(self.interval, 1)
        Q_cur = self.qos
        F_cur = self.fidelity

        # 8) true criticality on the NEXT state
        state_tuple = (int(dT_bin), int(bufT), int(B_bin), int(L_bin), int(D_bin),
                       int(I_cur), int(Q_cur), int(F_cur), int(anomT))
        true_crit = compute_true_criticality(state_tuple)

        payload = {
            "run_mode": RUN_MODE,
            "timestamp": time.time_ns(),

            # raw signals
            "temperature": round(float(temp), 3),
            "ema": round(float(self.temp_ema), 3),
            "deltaT": round(float(deltaT), 4),

            # battery/energy
            "battery_mAh": round(float(self.battery_mAh), 6),
            "battery_pct": round(float(self.battery_mAh / self.battery_capacity), 6),
            "energy_mAh_used": round(float(energy_used), 8),

            # state bins
            "dT": int(dT_bin), "bufT": int(bufT), "B": int(B_bin),
            "L": int(L_bin), "D": int(D_bin), "anomT": int(anomT),
            "I_cur": int(I_cur), "Q_cur": int(Q_cur), "F_cur": int(F_cur),

            # action snapshot (what was used to produce this point)
            "qos": int(self.qos), "interval_s": int(self.interval),
            "fidelity": int(self.fidelity), "critical": int(self.critical),

            # new: reliability & ground truth
            "packet_status": packet_status,
            "latency_ms": float(latency_ms),
            "true_critical": int(true_crit),
        }
        return payload


sensor = RealisticEnvSensor()

# ====== MQTT client for control actions ======


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        qos = int(data.get("qos", sensor.qos))
        interval = int(data.get("interval", data.get(
            "interval_s", sensor.interval)))
        fidelity = int(data.get("fidelity", sensor.fidelity))
        critical = int(data.get("critical", sensor.critical))
        sensor.apply_action(qos, interval, fidelity, critical)
        logging.info("Applied control: qos=%s interval=%s fidelity=%s critical=%s",
                     qos, interval, fidelity, critical)
    except Exception as e:
        logging.error(f"Control parse error: {e}")


def start_mqtt_control():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(CONTROL_TOPIC)
    client.loop_start()
    return client

# ====== CoAP resource ======


class SensorResource(resource.Resource):
    async def render_get(self, request):
        payload = json.dumps(sensor.step()).encode("utf-8")
        # 50 = application/json
        return aiocoap.Message(payload=payload, content_format=50)


async def main():
    logging.basicConfig(level=logging.INFO)
    start_mqtt_control()
    root = resource.Site()
    root.add_resource(['sensor'], SensorResource())
    await aiocoap.Context.create_server_context(root, bind=('0.0.0.0', 5683))
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())
