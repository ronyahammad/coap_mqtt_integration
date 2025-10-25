import os
import json
import time
import math
import queue
import random
import logging
from collections import deque, namedtuple
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import paho.mqtt.client as mqtt

# ===================== Config =====================
SUMMARY_TOPIC   = os.getenv("SUMMARY_TOPIC",   "UALG2025_THESIS_IOT_SENSORS/ddqn_episode_summary")
MQTT_BROKER     = os.getenv("MQTT_BROKER",     "mosquitto")
MQTT_PORT       = int(os.getenv("MQTT_PORT",   "1883"))
TELEMETRY_TOPIC = os.getenv("TELEMETRY_TOPIC", "UALG2025_THESIS_IOT_SENSORS/ddqn")
CONTROL_TOPIC   = os.getenv("CONTROL_TOPIC",   "UALG2025_THESIS_IOT_SENSORS/control")

# seed & training knobs
SEED         = int(os.getenv("SEED", "1337"))
LR           = float(os.getenv("LR", "3e-5"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "128"))
REPLAY_CAP   = int(os.getenv("REPLAY_CAP", "20000"))
TARGET_TAU   = float(os.getenv("TARGET_TAU", "0.01"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "4000"))
EPS_MIN      = float(os.getenv("EPS_MIN", "0.05"))
EPS_DECAY    = float(os.getenv("EPS_DECAY", "0.99997"))
SUMMARY_EVERY_STEPS = int(os.getenv("SUMMARY_EVERY_STEPS", "0"))  # 0 = disabled
SUMMARY_EVERY_STEPS = int(
    os.getenv("SUMMARY_EVERY_STEPS", "0"))  # 0 = disabled

def set_seed(s=SEED):
    np.random.seed(s); random.seed(s); torch.manual_seed(s)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Score = dT + bufT + (2-B) + L + D + anomT   in [0..11]
CRIT_THRESH = 4
def compute_true_criticality(state_tuple):
    dT, bufT, B, L, D, *_rest, anomT = state_tuple
    if int(anomT) == 1 or int(bufT) == 2 or int(L) == 2 or int(D) == 2:
        return 1
    score = int(dT) + int(bufT) + (2 - int(B)) + int(L) + int(D) + int(anomT)
    return 1 if score >= CRIT_THRESH else 0

# ===================== Replay Buffer (PER-light) =====================
class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.filled = 0

    def push(self, *args, td_error=1.0):
        i = self.pos
        self.buffer[i] = Transition(*args)
        self.priorities[i] = float(max(abs(td_error), 1e-5))
        self.pos = (i + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)

    def __len__(self): return self.filled

    def sample(self, batch_size, beta=0.4):
        n = self.filled
        idx = np.arange(n)
        p = np.maximum(self.priorities[:n] ** self.alpha, 1e-12); p /= p.sum()
        indices = np.random.choice(idx, size=batch_size, replace=(n < batch_size), p=p)
        weights = (n * p[indices]) ** (-beta); weights /= (weights.max() + 1e-12)
        batch = [self.buffer[i] for i in indices]
        return Transition(*zip(*batch)), indices, weights

    def update_priorities(self, indices, td_errors):
        self.priorities[np.asarray(indices, dtype=int)] = np.asarray(td_errors, dtype=np.float64) + 1e-5

# ===================== Reward =====================
class AdaptiveRewardCalculator:
    """
    Same structure as your offline reward, using energy from telemetry.
    """
    def __init__(self):
        self.w_I, self.w_T, self.w_R, self.w_C = 0.30, 0.20, 0.28, 0.30
        self.rho = 0.7; self.kappa = 2.0; self.e_ref = 0.010

    def calculate_reward(self, state, action, energy_consumed, true_crit):
        dT, bufT, B, L, D, *_rest, anomT = state
        qos, interval, fidelity, critical = action
        burst = 1 if (critical==1 and fidelity==2) else 0

        E = dT; Q = bufT; A = 1 if anomT else 0
        lambda_I = E/2 + A; lambda_T = E/2 + A; lambda_R = (L + D)/4; lambda_C = 1

        need = min(1.0, E/2.0 + A)
        cap  = (fidelity + 1)/3.0
        dem  = min(1.0, (E + Q)/3.0)
        U_info = min(1.0, cap * dem)
        U_time = (1.0 / (1.0 + interval/5.0 + D/2.0)) * need
        if E == 1 and B == 2 and A == 0 and L == 0 and D == 0:
            U_time *= 1.20

        g = 2 if (L==2 or D==2) else (1 if (L==1 or D==1) else 0)
        m = abs(qos - g)
        link_sev = 0.5 * (L/2.0 + D/2.0)
        U_rel = 1.0 - (m**2) * (0.6 + 0.8*link_sev)

        severity = min(2.0, 1.0 + 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
        U_crit = severity if (critical == true_crit) else -severity

        mu_E = 0.6 if B == 0 else 0.2
        mu_X = (L + D)/4
        mu_B = Q/2

        C_energy = min(1.0, energy_consumed/self.e_ref) * (1.0 + qos/2.0)
        payload_proxy = (fidelity + 1)/3.0
        C_net = payload_proxy * (L + D)/4.0

        C_backlog = (1.0 + interval/60.0) if (Q==2 and fidelity==0) else 0.0
        if (E == 2 or Q >= 1) and interval == 60:
            C_backlog += 0.35
        elif (E == 2 or Q >= 1) and interval == 30:
            C_backlog += 0.15
        if critical == 1 and Q == 2 and interval > 30:
            C_backlog += 0.20

        R_idle = self.rho if (E==0 and Q==0 and B==2 and qos==0 and interval==60 and fidelity==0 and critical==0) else 0.0
        eventy = 1 if (E==2 or Q==2 or A==1) else 0
        lazy   = 1 if (fidelity==0 or interval==60) else 0
        safety_pen = self.kappa if (eventy and lazy) else 0.0

        crit_fidelity_adj = 0.0
        if true_crit == 1 and critical == 1:
            target_f = 2 if (E == 2 or Q >= 1 or A == 1) else 1
            sev_f = min(1.0, 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
            scale = 1.0 if B >= 1 else 0.5
            if fidelity < target_f:
                crit_fidelity_adj -= scale * (0.15 + 0.25 * sev_f)
            elif fidelity == 2:
                crit_fidelity_adj += scale * (0.10 + 0.18 * sev_f)

        reward = ((self.w_I*lambda_I)*U_info +
                  (self.w_T*lambda_T)*U_time +
                  (self.w_R*lambda_R)*U_rel  +
                  (self.w_C*lambda_C)*U_crit
                  - mu_E*C_energy - mu_X*C_net - mu_B*C_backlog
                  + R_idle - safety_pen) + crit_fidelity_adj

        sev = min(1.0, 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
        bonus, penalty = (0.5 + 0.9*sev), (1.1 + 1.1*sev)
        if true_crit == 0 and critical == 1:
            reward -= 0.25 + 0.35*sev
        if true_crit == 1 and critical == 1:
            reward += bonus
        elif true_crit == 1 and critical == 0:
            reward -= penalty

        if eventy and (interval == 5 or (critical==1 and fidelity==2)):
            reward += 0.2

        return float(np.clip(reward / 4.0, -1.0, 1.0))

# ===================== Network =====================
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma0=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer('weight_eps', torch.zeros(out_f, in_f))
        self.register_buffer('bias_eps', torch.zeros(out_f))
        self.sigma0 = sigma0
        self.reset_parameters(); self.reset_noise()
    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_f))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_f))
    def reset_noise(self):
        eps_in  = torch.randn(self.in_f, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_f, device=self.weight_mu.device)
        f = lambda x: x.sign().mul_(x.abs().sqrt_())
        eps_in, eps_out = f(eps_in), f(eps_out)
        self.weight_eps.copy_(torch.ger(eps_out, eps_in)); self.bias_eps.copy_(eps_out)
    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

class DuelingNoisyDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.adv = NoisyLinear(64, action_size)
        self.val = NoisyLinear(64, 1)
        self.c_head = nn.Linear(64, 1)  # criticality head
    def reset_noise(self): self.adv.reset_noise(); self.val.reset_noise()
    def forward(self, x):
        h = F.relu(self.fc1(x)); h = F.relu(self.fc2(h))
        A = self.adv(h); V = self.val(h)
        q = V + A - A.mean(dim=1, keepdim=True)
        c = self.c_head(h)
        return q, c

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = (self.alpha*targets + (1-self.alpha)*(1-targets)) * (1-pt)**self.gamma * bce
        return loss.mean()

# ===================== Agent =====================
class DQNAgent:
    def __init__(self, actions):
        self.actions = actions
        self.action_size = len(actions)
        self.state_size = 25  # 8*3 one-hots + anom
        self.memory = ReplayBuffer(REPLAY_CAP)
        self.gamma = 0.99; self.n_step = 3; self.gamma_n = self.gamma ** self.n_step
        self._nstep_buf = deque(maxlen=self.n_step)
        self.epsilon = 1.0; self.epsilon_min = EPS_MIN; self.epsilon_decay = EPS_DECAY
        self.batch_size = BATCH_SIZE; self.tau = TARGET_TAU
        self.beta0, self.betaT = 0.4, 1.0
        self.total_steps = 0
        self.policy_net = DuelingNoisyDQN(self.state_size, self.action_size).to(device)
        self.target_net = DuelingNoisyDQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        # bias-init crit head to prior prevalence (15%)
        p0 = 0.15; b = float(np.log(p0/(1-p0)))
        with torch.no_grad():
            self.policy_net.c_head.bias.fill_(b)
            self.target_net.c_head.bias.fill_(b)
        self.q_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.crit_loss_fn = FocalLoss(alpha=0.9, gamma=2.0)
        self.reward_calculator = AdaptiveRewardCalculator()

    def one_hot_encode(self, state):
        encoded = []
        for v in state[:8]:
            v = int(v); v = v if v in (0,1,2) else 0
            encoded.extend(np.eye(3, dtype=np.float32)[v])
        encoded.append(float(state[8]))
        return np.array(encoded, dtype=np.float32)

    def get_state_tensor(self, state):
        return torch.from_numpy(self.one_hot_encode(state)).unsqueeze(0).to(device)

    def select_action(self, state, explore=True, training=True):
        if training and hasattr(self.policy_net, "reset_noise"): self.policy_net.reset_noise()
        st = self.get_state_tensor(state)
        with torch.no_grad():
            q_values_t, c_logit_t = self.policy_net(st)
            q_values = q_values_t.cpu().numpy().flatten()
            logit = float(c_logit_t.item())
            p_crit = 1.0 / (1.0 + math.exp(-logit))
        if explore and training:
            if random.random() < self.epsilon:
                pred_c = 1 if p_crit >= 0.35 else 0
                cand = [i for i, a in enumerate(self.actions) if a[3] == pred_c]
                if cand:
                    return random.choice(cand)
                return random.randrange(self.action_size)
            else:
                return int(np.argmax(q_values))
        else:
            return int(np.argmax(q_values))


    def predict_pcrit(self, state) -> float:
        st = torch.from_numpy(self.one_hot_encode(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, c_logit_t = self.policy_net(st)
            logit = float(c_logit_t.item())
            return 1.0 / (1.0 + math.exp(-logit))

    def store_transition(self, s, a_idx, r, ns, done, true_crit):
        self._nstep_buf.append((s, a_idx, r, ns, done))
        if len(self._nstep_buf) < self.n_step: return
        R, ns_n, d = 0.0, self._nstep_buf[-1][3], self._nstep_buf[-1][4]
        for k, (_, _, rk, _, dk) in enumerate(self._nstep_buf):
            R += (self.gamma ** k) * rk
            if dk: d = True; break
        s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
        seed = 1.0 + 4.0*true_crit
        self.memory.push(s0, a0, R, ns_n, d, td_error=seed)

    def flush_nstep(self):
        while len(self._nstep_buf) > 0:
            R, ns_n, d = 0.0, self._nstep_buf[-1][3], self._nstep_buf[-1][4]
            for k, (_, _, rk, _, dk) in enumerate(self._nstep_buf):
                R += (self.gamma ** k) * rk
                if dk: d = True; break
            s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
            self.memory.push(s0, a0, R, ns_n, d, td_error=1.0)
            self._nstep_buf.popleft()

    def learn(self):
        if len(self.memory) < self.batch_size: return None
        frac = min(1.0, self.total_steps / max(1, WARMUP_STEPS))
        beta = self.beta0 + (self.betaT - self.beta0) * frac
        batch, indices, weights = self.memory.sample(self.batch_size, beta=beta)

        state_batch = torch.from_numpy(np.array([self.one_hot_encode(s) for s in batch.state], dtype=np.float32)).to(device)
        next_state_batch = torch.from_numpy(np.array([self.one_hot_encode(s) for s in batch.next_state], dtype=np.float32)).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(device)

        q_pred, c_logit = self.policy_net(state_batch)
        current_q = q_pred.gather(1, action_batch)

        with torch.no_grad():
            next_q_policy, _ = self.policy_net(next_state_batch)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target, _ = self.target_net(next_state_batch)
            next_q = next_q_target.gather(1, next_actions)
            target_q = reward_batch + (1 - done_batch) * (self.gamma ** 3) * next_q

        td_per_elem = F.smooth_l1_loss(current_q, target_q, reduction='none')
        q_loss = (td_per_elem * weights_t).mean()

        crit_labels = torch.FloatTensor([compute_true_criticality(s) for s in batch.state]).unsqueeze(1).to(device)
        crit_loss = self.crit_loss_fn(c_logit, crit_labels)

        loss = q_loss + 1.0 * crit_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            td_error = (current_q - target_q).abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_error)

        # soft target update
        with torch.no_grad():
            tau = self.tau
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * pp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1
        return float(loss.item())

# ===================== MQTT Agent loop =====================
def build_state_from_payload(d: Dict[str, Any]) -> Tuple[int,int,int,int,int,int,int,int,int]:
    dT   = int(d.get("dT", 0))
    bufT = int(d.get("bufT", 0))
    B    = int(d.get("B", 2))
    L    = int(d.get("L", 0))
    D    = int(d.get("D", 0))
    Icur = int(d.get("I_cur", 1))
    Qcur = int(d.get("Q_cur", int(d.get("qos", 0))))
    Fcur = int(d.get("F_cur", int(d.get("fidelity", 0))))
    anom = int(d.get("anomT", 0))
    return (dT, bufT, B, L, D, Icur, Qcur, Fcur, anom)

def make_actions():
    intervals = [5, 30, 60]
    return [(q, t, f, c)
            for q in [0, 1, 2]
            for t in intervals
            for f in [0, 1, 2]
            for c in [0, 1]]

class PolicyRunner:
    def __init__(self):
        self.actions = make_actions()
        self.agent = DQNAgent(self.actions)
        self.telemetry_q: "queue.Queue[Dict[str,Any]]" = queue.Queue(maxsize=1000)
        self.last_state: Optional[Tuple[int,...]] = None
        self.last_action_idx: Optional[int] = None
        self.mqtt = None
        # episode accumulators (for ddqn_episode_summary)
        self.steps = 0
        self.sum_energy = 0.0
        self.sum_latency = 0.0
        self.sum_reward = 0.0
        self.on_time = 0
        self.late = 0
        self.drop = 0
        self.episode = 0

    # MQTT callbacks
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logging.info("Connected to MQTT (rc=%s). Subscribing to %s", rc, TELEMETRY_TOPIC)
            client.subscribe(TELEMETRY_TOPIC, qos=1)
        else:
            logging.error("MQTT connect failed: rc=%s", rc)

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            if isinstance(data, dict):
                self.telemetry_q.put_nowait(data)
        except Exception as e:
            logging.error("Telemetry parse error: %s", e)

    def start_mqtt(self):
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.on_connect = self._on_connect
        c.on_message = self._on_message
        backoff = 1
        while True:
            try:
                c.connect(MQTT_BROKER, MQTT_PORT, 60)
                c.loop_start()
                self.mqtt = c
                return
            except Exception as e:
                logging.warning("MQTT connect failed: %s. Retrying in %ss...", e, backoff)
                time.sleep(backoff); backoff = min(30, 2*backoff)

    def publish_action(self, a, p_crit: float = None):
        payload = {
            "qos": a[0],
            "interval": a[1],     # bridge normalizes to interval_s
            "fidelity": a[2],
            "critical": a[3],
            "timestamp": int(time.time() * 1e9),
        }
        if p_crit is not None:
            payload["p_crit"] = float(p_crit)
        self.mqtt.publish(CONTROL_TOPIC, json.dumps(payload))
        logging.info("→ Action: %s", payload)


    def _emit_episode_summary(self):
        if self.steps == 0: return
        summary = {
            "episode": self.episode,
            "steps": self.steps,
            "total_reward": round(self.sum_reward, 6),
            "total_energy_mAh": round(self.sum_energy, 6),
            "avg_latency_ms": (self.sum_latency / max(1, self.steps)),
            "on_time_rate": self.on_time / max(1, self.steps),
            "late_rate": self.late / max(1, self.steps),
            "drop_rate": self.drop / max(1, self.steps),
            "timestamp": int(time.time() * 1e9),
        }
        self.mqtt.publish(SUMMARY_TOPIC, json.dumps(summary))
        logging.info("↳ Episode summary published: %s", summary)
        # reset counters for next episode
        self.episode += 1
        self.steps = self.on_time = self.late = self.drop = 0
        self.sum_energy = self.sum_latency = self.sum_reward = 0.0

    def _emit_periodic_snapshot(self):
        if self.steps == 0:
            return
        snap = {
            "episode": self.episode,  # current episode index
            "steps": self.steps,
            "total_reward": round(self.sum_reward, 6),
            "total_energy_mAh": round(self.sum_energy, 6),
            "avg_latency_ms": (self.sum_latency / max(1, self.steps)),
            "on_time_rate": self.on_time / max(1, self.steps),
            "late_rate": self.late / max(1, self.steps),
            "drop_rate": self.drop / max(1, self.steps),
            "timestamp": int(time.time() * 1e9),
            # Optional marker so you can distinguish if you want (Influx will ignore unknown fields):
            "kind": 1  # 1 = periodic, 0/absent = end-of-episode
        }
        self.mqtt.publish(SUMMARY_TOPIC, json.dumps(snap))


    def run(self):
        self.start_mqtt()
        logging.info("Policy runner started. Waiting for telemetry...")
        while True:
            d = self.telemetry_q.get()  # blocking
            try:
                # optional counters (if provided by server)
                pkt = d.get("packet_status")
                if pkt == "on_time": self.on_time += 1
                elif pkt == "late": self.late += 1
                elif pkt == "dropped": self.drop += 1
                self.sum_energy += float(d.get("energy_mAh_used", 0.0))
                self.sum_latency += float(d.get("latency_ms", 0.0))
                self.steps += 1

                if SUMMARY_EVERY_STEPS > 0 and (self.steps % SUMMARY_EVERY_STEPS == 0):
                    self._emit_periodic_snapshot()

                # 1) Build NEXT state from telemetry
                next_state = build_state_from_payload(d)
                energy_used = float(d.get("energy_mAh_used", 0.0))
                battery_mAh = float(d.get("battery_mAh", 0.0))
                done = battery_mAh <= 0.0
                true_crit = compute_true_criticality(next_state)

                # 2) If we had a previous state & action, compute reward and learn
                if self.last_state is not None and self.last_action_idx is not None:
                    prev_state = self.last_state
                    prev_action = self.actions[self.last_action_idx]
                    reward = self.agent.reward_calculator.calculate_reward(prev_state, prev_action, energy_used, true_crit)
                    self.sum_reward += reward
                    self.agent.store_transition(prev_state, self.last_action_idx, reward, next_state, done, true_crit)
                    loss = self.agent.learn()
                    if loss is not None and (self.agent.total_steps % 100 == 0):
                        logging.info("[learn] step=%d loss=%.5f eps=%.3f", self.agent.total_steps, loss, self.agent.epsilon)

                # 3) Select new action based on NEXT state
                a_idx = self.agent.select_action(
                    next_state, explore=True, training=True)
                action = self.actions[a_idx]
                p_crit = self.agent.predict_pcrit(next_state)
                self.publish_action(action, p_crit=p_crit)

                # 4) Update trackers
                self.last_state = next_state
                self.last_action_idx = a_idx

                # 5) If battery is dead, flush buffers and emit summary
                if done:
                    self.agent.flush_nstep()
                    self._emit_episode_summary()
                    self.last_state = None
                    self.last_action_idx = None
                    logging.info("Battery depleted — waiting for next telemetry (new episode).")

            except Exception as e:
                logging.error("Policy loop error: %s", e, exc_info=True)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    runner = PolicyRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        logging.info("Policy agent interrupted; exiting.")

if __name__ == "__main__":
    main()
