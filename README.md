# Publication title : Learning-Based Adaptive Transmission Control in Hybrid CoAP-MQTT IoT Systems

Accepted in IEEE ISCC 2026 conference
---
# Thesis Title : Optimization of Data Transmission in IoT Applications using CoAP and MQTT protocols

CoAP ⇄ MQTT ⇄ InfluxDB ⇄ Grafana + DDQN Policy Agent

This repository contains a full Dockerised stack:

* **CoAP sensor server** (simulated temperature sensor + battery/backlog model) 
* **CoAP → MQTT bridge** (pulls CoAP JSON, publishes to MQTT by run mode) 
* **DDQN policy agent** (subscribes to telemetry, learns a control policy, publishes actions + episode summaries) 
* **MQTT → InfluxDB bridge** (writes telemetry, actions, and episode summaries into InfluxDB) 
* **Mosquitto MQTT broker** (Dockerised, open for local connections) 
* **InfluxDB + Grafana** for storage and dashboards
* **MQTT subscriber** to inspect all MQTT traffic from all pipelines 

Everything is started from `docker-compose.yml`.

---

## 1. Prerequisites

* Docker Desktop (Windows/Mac) or Docker Engine (Linux)
* Docker Compose (comes with Docker Desktop)
* Git (to clone the repo)
* A web browser (for InfluxDB + Grafana UI)

---

## 2. Clone the repository and start the stack

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/coap_mqtt_integration.git
cd coap_mqtt_integration

# 2) Start all services (build images on first run)
docker-compose up -d --build
```

What this does:

* Builds all Python services with the correct Python version and libraries inside containers.
* Starts:

  * `coap_server`
  * `coap_mqtt_bridge`
  * `ddqn_policy_agent`
  * `mqtt_influx_bridge`
  * `mqtt_subscriber`
  * `mosquitto`
  * `influxdb`
  * `grafana`

In Docker Desktop you should see a **project** called `coap_mqtt_integration` with multiple running containers.

---

## 3. Check that MQTT traffic is flowing

The **mqtt_subscriber** container subscribes to `UALG2025_THESIS_IOT_SENSORS/#` and prints all non-control messages. 

```bash
# Show live logs from the subscriber
docker logs -f coap_mqtt_integration_mqtt_subscriber_1
```

You should see JSON payloads like:

* `UALG2025_THESIS_IOT_SENSORS/no_rl` or `.../ddqn` (sensor telemetry)
* `UALG2025_THESIS_IOT_SENSORS/ddqn_episode_summary` (episode summaries)
* `UALG2025_THESIS_IOT_SENSORS/control` (actions sent by the agent)

If that works, the CoAP server, bridge, policy agent and Mosquitto are talking to each other correctly.

---

## 4. First-time InfluxDB setup

### 4.1 Log into InfluxDB

Open in your browser:

```text
http://localhost:8086/signin
```

Default credentials (from the stack):

* **Username**: `admin`
* **Password**: `admin123`

> If you are prompted to create the initial user/org/bucket, follow the wizard with:
>
> * org: `iot`
> * bucket: `iot_data`

### 4.2 Get the admin token and put it in `.env`

In InfluxDB UI:

1. Go to **Load Data → Sources → Influx CLI → Initialize Client**.

2. Copy the `--token` string (long random token).

3. In the **project root**, open `.env` (create it if it does not exist) and set:

   ```env
   INFLUXDB_INIT_ADMIN_TOKEN=PASTE_YOUR_TOKEN_HERE
   # If there is INFLUX_TOKEN in .env, set the same value there as well
   INFLUX_TOKEN=PASTE_YOUR_TOKEN_HERE
   ```

   The MQTT→Influx bridge uses `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_BUCKET`, and `INFLUX_TOKEN` to write into your Influx instance. 

4. Restart the stack so the new token is picked up:

   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

5. Log into InfluxDB again at `http://localhost:8086/signin` with the same `admin / admin123`.

At this point, the `mqtt_influx_bridge` should be writing data into bucket **`iot_data`** in org **`iot`**:

* Measurement `ddqn_episode_summary` for episode summaries. 
* Measurement `control_actions` for all actions sent by the agent. 
* Measurements `sensor_data` (baseline, `no_rl`) and `ddqn_data` (adaptive, `ddqn`) for telemetry.

---

## 5. Configure Grafana

### 5.1 Log into Grafana

Open:

```text
http://localhost:3000
```

Default Grafana credentials (if not changed):

* **Username**: `admin`
* **Password**: `admin`

Grafana is already pointed at InfluxDB in the compose setup (if your stack is configured that way). If not, add the data source manually.

### 5.2 Add InfluxDB data source in Grafana (if needed)

1. In Grafana left sidebar, click **“Connections → Data sources”**.
2. Add **InfluxDB**.
3. Set:

   * **URL**: `http://influxdb:8086`
   * **Organization**: `iot`
   * **Token**: paste the same token from `.env` (`INFLUXDB_INIT_ADMIN_TOKEN` / `INFLUX_TOKEN`).
   * **Default bucket**: `iot_data`
4. Click **Save & Test** – it should succeed.

---

## 6. Grafana dashboard variables (important for regex)

Some panels use Grafana variables such as `${mode:regex}` and `${win:raw}` in the Flux queries. If these variables do not exist, queries will fail.

### 6.1 Create `mode` variable (run_mode regex)

1. Open any dashboard (or create a new one).
2. Click **Dashboard settings → Variables → Add variable**.
3. Name: `mode`
4. Type: **Custom**
5. Values: `no_rl,ddqn`
6. Enable **Multi-value** and **Include All** (optional).

Grafana will automatically expand `${mode:regex}` into a regex like `no_rl|ddqn`.

### 6.2 Create `win` variable (aggregation window)

1. Add another variable:
2. Name: `win`
3. Type: **Custom**
4. Values: `5m,10m,30m,1h` (or any list of windows you prefer)
5. Default value: `5m`

Then `${win:raw}` will interpolate as `5m`, `10m`, etc.

### 6.3 (Optional) Create interval regex variable for 5s / 30s / 60s

If you want panels that filter by interval, you can create:

1. Name: `interval`
2. Type: **Custom**
3. Values: `5|30|60`

and then in Flux use:

```flux
|> filter(fn: (r) => string(v: r.interval_s) =~ /${interval:regex}/)
```

This is where the “regex for 5s, 30s, 60s” comes in.

---

## 7. Measurements written by the MQTT→Influx bridge

For reference (this matters when building queries): 

* **`ddqn_episode_summary`**

  * Tag:

    * `run_mode = "ddqn"`
  * Fields:

    * `episode`, `steps`
    * `total_reward`
    * `total_energy_mAh`
    * `avg_latency_ms`
    * `on_time_rate`, `late_rate`, `drop_rate`

* **`control_actions`**

  * Fields:

    * `interval_s`, `qos`, `fidelity`, `critical`
    * `p_crit` (predicted probability of criticality)

* **`sensor_data`** (baseline / no RL)
  **`ddqn_data`** (adaptive / DDQN)

  * Tags:

    * `run_mode ∈ {"no_rl","ddqn"}`
    * `packet_status ∈ {"on_time","late","dropped"}`
  * Fields:

    * `latency_ms`
    * `true_critical`
    * `temperature`, `ema`, `deltaT`
    * `battery_mAh`, `battery_pct`, `energy_mAh_used`
    * Binned state variables: `dT, bufT, B, L, D, anomT, I_cur, Q_cur, F_cur`
    * Action snapshot: `qos, interval_s, fidelity, critical`

---

## 8. Example Grafana panels and Flux queries

Below are ready-to-paste Flux queries for typical panels.
For each panel:

* **From**: InfluxDB data source you created
* **Bucket**: `iot_data`

### 8.1 Return per episode (true value)

**Panel type**: Time series (episode on X, total reward on Y)

```flux
from(bucket: "iot_data")
  |> range(start: 0, stop: now())
  |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
  |> filter(fn: (r) => r._field == "total_reward")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "episode", "total_reward"])
  |> group(columns: [])
  |> sort(columns: ["episode"])
  |> map(fn: (r) => ({
      _time: r._time,
      _value: r.total_reward,
      episode: r.episode,
    }))
```

---

### 8.2 Two-window return (first 100 vs last 100) + ΔR and % gain

**Panel type**: SingleStat / Table

```flux
import "math"

base =
  from(bucket: "iot_data")
    |> range(start: 0, stop: now())
    |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "episode", "total_reward"])
    |> group()
    |> sort(columns: ["episode"])

w_first =
  base
    |> limit(n: 100)
    |> mean(column: "total_reward")
    |> map(fn: (r) => ({ _time: now(), window: "first100", val: r.total_reward }))

w_last =
  base
    |> tail(n: 100)
    |> mean(column: "total_reward")
    |> map(fn: (r) => ({ _time: now(), window: "last100", val: r.total_reward }))

union(tables: [w_first, w_last])
  |> pivot(rowKey: ["_time"], columnKey: ["window"], valueColumn: "val")
  |> map(fn: (r) => ({
      _time: r._time,
      R_1_100: r.first100,
      R_last100: r.last100,
      dR: r.last100 - r.first100,
      rel_gain_pct: if r.first100 != 0.0 then
        (r.last100 - r.first100) / math.abs(x: r.first100) * 100.0
      else
        0.0,
    }))
```

---

### 8.3 On-time rate per episode

```flux
from(bucket: "iot_data")
  |> range(start: 0, stop: now())
  |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
  |> filter(fn: (r) => r._field == "on_time_rate")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "episode", "on_time_rate"])
  |> group(columns: [])
  |> sort(columns: ["episode"])
  |> map(fn: (r) => ({
      _time: r._time,
      _value: r.on_time_rate,
      episode: r.episode,
    }))
```

---

### 8.4 Two-window On-time rate (percentage points & % improvement)

```flux
import "math"

base =
  from(bucket: "iot_data")
    |> range(start: 0, stop: now())
    |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "episode", "on_time_rate"])
    |> group()
    |> sort(columns: ["episode"])

w_first =
  base
    |> limit(n: 100)
    |> mean(column: "on_time_rate")
    |> map(fn: (r) => ({ _time: now(), window: "first100", val: r.on_time_rate }))

w_last =
  base
    |> tail(n: 100)
    |> mean(column: "on_time_rate")
    |> map(fn: (r) => ({ _time: now(), window: "last100", val: r.on_time_rate }))

union(tables: [w_first, w_last])
  |> pivot(rowKey: ["_time"], columnKey: ["window"], valueColumn: "val")
  |> map(fn: (r) => ({
      _time: r._time,
      OnTime_1_100: r.first100,
      OnTime_last100: r.last100,
      d_pp: (r.last100 - r.first100) * 100.0, // percentage points
      rel_impr_pct: if r.first100 > 0.0 then
        (r.last100 - r.first100) / r.first100 * 100.0
      else
        0.0,
    }))
```

---

### 8.5 Energy per episode (true value)

```flux
from(bucket: "iot_data")
  |> range(start: 0, stop: now())
  |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
  |> filter(fn: (r) => r._field == "total_energy_mAh")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "episode", "total_energy_mAh"])
  |> group(columns: [])
  |> sort(columns: ["episode"])
  |> map(fn: (r) => ({
      _time: r._time,
      _value: r.total_energy_mAh,
      episode: r.episode,
    }))
```

---

### 8.6 Two-window energy saving `s` and lifetime factor `F`

```flux
import "math"

base =
  from(bucket: "iot_data")
    |> range(start: 0, stop: now())
    |> filter(fn: (r) => r._measurement == "ddqn_episode_summary")
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "episode", "total_energy_mAh"])
    |> group()
    |> sort(columns: ["episode"])

w_first =
  base
    |> limit(n: 100)
    |> mean(column: "total_energy_mAh")
    |> map(fn: (r) => ({ _time: now(), window: "first100", val: r.total_energy_mAh }))

w_last =
  base
    |> tail(n: 100)
    |> mean(column: "total_energy_mAh")
    |> map(fn: (r) => ({ _time: now(), window: "last100", val: r.total_energy_mAh }))

union(tables: [w_first, w_last])
  |> pivot(rowKey: ["_time"], columnKey: ["window"], valueColumn: "val")
  |> map(fn: (r) => ({
      _time: r._time,
      E_1_100: r.first100,
      E_last100: r.last100,
      s: if r.first100 > 0.0 then
        (r.first100 - r.last100) / r.first100
      else
        0.0,
      F: if r.first100 > 0.0 then
        1.0 / (1.0 - ((r.first100 - r.last100) / r.first100))
      else
        1.0,
    }))
```

---

### 8.7 Latency p50 / p95 panel

**Panel type**: Time series (two series, p50 and p95).
Uses variables: `${mode:regex}`, `${win:raw}`.

```flux
p95 =
  from(bucket: "iot_data")
    |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
    |> filter(fn: (r) =>
      r._measurement =~ /^(sensor_data|ddqn_data)$/ and
      r.run_mode =~ /${mode:regex}/ and
      r._field == "latency_ms"
    )
    |> group(columns: ["run_mode"])
    |> aggregateWindow(
      every: ${win:raw},
      fn: (column, tables=<-) =>
        quantile(tables: tables, q: 0.95, column: column, method: "estimate_tdigest"),
      createEmpty: false
    )
    |> set(key: "quantile", value: "p95")

p50 =
  from(bucket: "iot_data")
    |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
    |> filter(fn: (r) =>
      r._measurement =~ /^(sensor_data|ddqn_data)$/ and
      r.run_mode =~ /${mode:regex}/ and
      r._field == "latency_ms"
    )
    |> group(columns: ["run_mode"])
    |> aggregateWindow(
      every: ${win:raw},
      fn: (column, tables=<-) =>
        quantile(tables: tables, q: 0.50, column: column, method: "estimate_tdigest"),
      createEmpty: false
    )
    |> set(key: "quantile", value: "p50")

union(tables: [p95, p50])
```

---

### 8.8 Late & drop rates (stacked area)

This builds late/drop rates over time using the `packet_status` tag.

```flux
// Late rate
late =
  from(bucket: "iot_data")
    |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
    |> filter(fn: (r) =>
      r._measurement =~ /^(sensor_data|ddqn_data)$/ and
      r.run_mode =~ /${mode:regex}/ and
      r._field == "latency_ms"
    )
    |> map(fn: (r) => ({
      r with _value:
        if r.packet_status == "late" then 1.0 else 0.0,
    }))
    |> aggregateWindow(every: ${win:raw}, fn: mean, createEmpty: false)
    |> rename(columns: { _value: "late_rate" })

// Drop rate
drop =
  from(bucket: "iot_data")
    |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
    |> filter(fn: (r) =>
      r._measurement =~ /^(sensor_data|ddqn_data)$/ and
      r.run_mode =~ /${mode:regex}/ and
      r._field == "latency_ms"
    )
    |> map(fn: (r) => ({
      r with _value:
        if r.packet_status == "dropped" then 1.0 else 0.0,
    }))
    |> aggregateWindow(every: ${win:raw}, fn: mean, createEmpty: false)
    |> rename(columns: { _value: "drop_rate" })

union(tables: [late, drop])
```

Use a **stacked area** visualization with `late_rate` and `drop_rate`.

---

### 8.9 QoS × interval bar chart (action counts)

Counts how many actions of each `(qos, interval_s)` pair were issued.

```flux
from(bucket: "iot_data")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "control_actions")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "interval_s", "qos"])
  |> map(fn: (r) => ({ r with one: 1.0 })) // 1 per action event
  |> group(columns: ["interval_s", "qos"])
  |> sum(column: "one")                      // count actions per (interval_s, qos)
  |> rename(columns: { one: "count" })
  |> pivot(rowKey: ["interval_s"], columnKey: ["qos"], valueColumn: "count")
```

Use a **bar chart** (or heatmap) where:

* X axis: `qos`
* Y axis: `interval_s` (5, 30, 60)
* Value: `count`

---

## 9. Switching between baseline and DDQN modes

The CoAP server emits telemetry with a `run_mode` field (e.g., `no_rl` or `ddqn`). 
The MQTT→Influx bridge uses this to route data into `sensor_data` vs `ddqn_data`. 

* To run **baseline** (no RL control), set `RUN_MODE=no_rl` for the `coap_server` service in `docker-compose.yml` or in `.env`, and stop the `ddqn_policy_agent` service if you don’t want actions.
* To run **adaptive DDQN**, set `RUN_MODE=ddqn` and ensure `ddqn_policy_agent` is running.

Then you can compare panels for `run_mode = "no_rl"` vs `run_mode = "ddqn"` using the `mode` variable (or by hard-coding filters in Flux).

