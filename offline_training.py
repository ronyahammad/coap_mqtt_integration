# step 6 for online fit for increses episode 700 and max_steps 250
# add one more “moderate ΔT with good battery” rule since i  want shorter intervals in those cases.
# Considering slightly stronger shaping toward fidelity=2 in ideal-critical cases to close the small gap.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ----- Setup -----
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def set_seed(seed: int = 1337):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Criticality (T-only) ---
# Score = dT + bufT + (2-B) + L + D + anomT   in [0..11]
CRIT_THRESH = 4  # will be gently nudged by the auto-calibrator if prevalence is off

def compute_true_criticality(state):
    dT, bufT, B, L, D, *_rest, anomT = state
    # Hard triggers match safety override
    if int(anomT) == 1 or int(bufT) == 2 or int(L) == 2 or int(D) == 2:
        return 1
    score = int(dT) + int(bufT) + (2 - int(B)) + int(L) + int(D) + int(anomT)
    return 1 if score >= CRIT_THRESH else 0


# ----- Environment (T-only) -----
class RealisticEnvSimulator:
    """
    Added knobs:
      - anom_threshold (default 2.5) can be decreased to raise positives
      - a_scale multiplies accumulation a_values to increase backlog pressure
    """
    def __init__(self, anom_threshold: float = 2.5, a_scale: float = 1.0):
        self.t = 0
        # temperature signal
        self.period = 288; self.ampl = 10; self.bias = 20; self.noise_std = 0.2

        # EMA
        self.temp_ema = None
        self.delta_alpha = 0.1
        self.delta_std = 1.0

        # Battery
        self.battery_capacity = 1000
        self.battery_mAh = self.battery_capacity

        # Sliding windows
        self.window_size = 20
        self.delay_window = deque(maxlen=self.window_size)
        self.loss_window = deque(maxlen=self.window_size)

        # Queue / anomaly
        self.backlog_T = 0
        self.capacity = 300
        self.anom_threshold = float(anom_threshold)

        # Constants
        base_a = [0.2, 0.6, 1.2]
        self.a_values = [a * float(a_scale) for a in base_a]      # accumulation
        self.kappa_F = [0.5, 1.0, 2.0]       # drain factor
        self.phi_f   = [0.6, 1.0, 1.6]
        self.overhead_q = [1.0, 1.15, 1.35]
        self.I_tx_base = 120  # mA
        self.I_base = 5       # mA
        self.I_sense = 2      # mA
        self.t_tx_base = 0.1  # s
        self.beta_f = 12
        self.epsilon_f = 0.015

    def temp_wave(self):
        return (self.bias
                + self.ampl * np.sin(2 * np.pi * self.t / self.period)
                + np.random.normal(0, self.noise_std))

    def update_ema(self, ema, x):
        return x if ema is None else self.delta_alpha * x + (1 - self.delta_alpha) * ema

    def compute_deltaT(self, temp):
        self.temp_ema = self.update_ema(self.temp_ema, temp)
        return abs(temp - self.temp_ema) / (self.delta_std + 1e-8)

    def bin_value(self, value, thresholds):
        if value < thresholds[0]: return 0
        elif value < thresholds[1]: return 1
        return 2

    def update_backlog(self, delta_T, action, burst):
        # accumulate & drain a single temperature queue
        self.backlog_T = min(self.capacity, self.backlog_T + self.a_values[delta_T])
        qos, interval, fidelity, _ = action
        drain_cap = self.kappa_F[fidelity] * (5 / interval) * (1 + burst)
        drain = min(self.backlog_T, drain_cap)
        self.backlog_T -= drain
        ratio_T = self.backlog_T / self.capacity
        buf_T = 0 if ratio_T < 1/3 else (1 if ratio_T < 2/3 else 2)
        return buf_T

    def update_battery(self, action, burst):
        qos, interval, fidelity, _ = action
        duty = min(1, (self.t_tx_base * self.phi_f[fidelity]) / interval * (1 + burst))
        I_tx = self.I_tx_base * self.overhead_q[qos] * duty
        I_avg = self.I_base + self.I_sense + I_tx
        k = 1.07
        delta_mAh = (I_avg**k) / 3600
        self.battery_mAh = max(0, self.battery_mAh - delta_mAh)
        battery = self.battery_mAh / self.battery_capacity
        B = 0 if battery < 0.3 else (1 if battery < 0.7 else 2)
        return B, delta_mAh

    def compute_loss_rate(self):
        if len(self.loss_window) == 0: return 0.0
        return 1.0 - (sum(self.loss_window) / len(self.loss_window))

    def compute_avg_delay(self):
        if len(self.delay_window) == 0: return 0.0
        return sum(self.delay_window) / len(self.delay_window)

    def simulate_environment(self, state, action):
        # state = (dT, bufT, B, L, D, I_cur, Q_cur, F_cur, anomT)
        dT, bufT, B, L, D, I_cur, Q_cur, F_cur, anomT = state

        # 1) new reading
        temp = self.temp_wave(); self.t += 1
        deltaT = self.compute_deltaT(temp)
        dT = self.bin_value(deltaT, [1, 3])

        # 2) burst if crit&high-fidelity
        _, _, fidelity, critical = action
        burst = 1 if (critical == 1 and fidelity == 2) else 0

        # 3) backlog & battery
        bufT = self.update_backlog(dT, action, burst)
        B, energy_consumed = self.update_battery(action, burst)

        # 4) network
        qos, interval, fidelity, _ = action
        base_delay = np.random.normal(100 + 50*(2-qos), 20)
        delay = np.clip(base_delay + 1000/interval + self.beta_f*fidelity, 0, 800)
        self.delay_window.append(delay)
        loss_chance = 0.18*(2 - qos) + 0.008*delay/100 + 0.012*fidelity

        is_delivered = np.random.rand() > loss_chance
        self.loss_window.append(int(is_delivered))

        # 5) loss/delay bins over sliding window
        loss_rate = self.compute_loss_rate()
        L = self.bin_value(loss_rate, [0.1, 0.3])
        avg_delay = self.compute_avg_delay()
        D = self.bin_value(avg_delay, [100, 500])

        packet_status = 'dropped'
        if is_delivered:
            packet_status = 'on_time' if delay < 300 else 'late'
        latency = delay if is_delivered else 1000

        # anomaly on temperature
        anomT = 1 if deltaT > self.anom_threshold else 0

        # reflect chosen controls
        I_cur = {5:0, 30:1, 60:2}[interval]
        Q_cur = qos
        F_cur = fidelity

        next_state = (dT, bufT, B, L, D, I_cur, Q_cur, F_cur, anomT)
        true_crit = compute_true_criticality(next_state)
        return next_state, packet_status, true_crit, energy_consumed, latency


# ----- Replay Buffer (PER) with stratified positives by CURRENT STATE -----
class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = [None] * capacity

        # priorities (PER)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        # labels for current state and next state
        self.labels_s  = np.full(capacity, -1, dtype=np.int8)  # current-state label
        self.labels_ns = np.full(capacity, -1, dtype=np.int8)  # next-state label

        self.filled = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0

        # index sets for current-state label
        self.pos_idx_s, self.neg_idx_s = set(), set()

        # counts
        self.n_pos_s = 0
        self.n_filled = 0

    @property
    def prevalence(self):
        return self.n_pos_s / max(1, self.n_filled)

    def _update_sets_on_overwrite(self, i):
        if not self.filled[i]: return
        old = self.labels_s[i]
        if old == 1:
            self.pos_idx_s.discard(i); self.n_pos_s -= 1
        elif old == 0:
            self.neg_idx_s.discard(i)

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        i = self.pos
        self._update_sets_on_overwrite(i)

        self.buffer[i] = Transition(state, action, reward, next_state, done)
        self.priorities[i] = float(max(td_error, 1e-5))

        lbl_s  = compute_true_criticality(state)
        lbl_ns = compute_true_criticality(next_state)
        self.labels_s[i]  = lbl_s
        self.labels_ns[i] = lbl_ns

        if lbl_s == 1:
            self.pos_idx_s.add(i); self.n_pos_s += 1
        else:
            self.neg_idx_s.add(i)

        if not self.filled[i]:
            self.filled[i] = True; self.n_filled += 1

        self.pos = (i + 1) % self.capacity

    def sample_stratified(self, batch_size, beta=0.4, pos_frac=0.40):
        filled_idx = np.flatnonzero(self.filled)
        pr = np.maximum(self.priorities[filled_idx] ** self.alpha, 1e-12)
        probs_all = pr / pr.sum()

        pos_pool = np.array(sorted(self.pos_idx_s), dtype=int)
        neg_pool = np.array(sorted(self.neg_idx_s), dtype=int)

        npos = max(1, int(batch_size * pos_frac))
        nneg = batch_size - npos

        def _pick(pool, n):
            if pool.size == 0: return np.empty(0, dtype=int)
            w = np.maximum(self.priorities[pool] ** self.alpha, 1e-12); w /= w.sum()
            return np.random.choice(pool, n, replace=(pool.size < n), p=w)

        idx_pos = _pick(pos_pool, npos)
        idx_neg = _pick(neg_pool, nneg)
        indices = np.concatenate([idx_pos, idx_neg])

        if indices.size < batch_size:
            rest = np.setdiff1d(filled_idx, indices, assume_unique=False)
            need = batch_size - indices.size
            add = np.random.choice(rest if rest.size else filled_idx, need, replace=(rest.size < need))
            indices = np.concatenate([indices, add])

        batch = [self.buffer[i] for i in indices]

        map_to_filled = {j: k for k, j in enumerate(filled_idx)}
        p = np.array([probs_all[map_to_filled[i]] for i in indices])
        w = (self.n_filled * p) ** (-beta); w /= (w.max() + 1e-12)

        return Transition(*zip(*batch)), indices, w

    def update_priorities(self, indices, td_errors):
        self.priorities[np.asarray(indices, dtype=int)] = np.asarray(td_errors, dtype=np.float64) + 1e-5

    def __len__(self):
        return self.n_filled


# ----- Reward (T-only, rebalanced) -----
class AdaptiveRewardCalculator:
    def __init__(self):
        self.w_I, self.w_T, self.w_R, self.w_C = 0.30, 0.20, 0.28, 0.30
        self.rho = 0.7; self.kappa = 2.0; self.e_ref = 0.010

    def calculate_reward(self, state, action, energy_consumed, true_crit):
        dT, bufT, B, L, D, *_rest, anomT = state
        qos, interval, fidelity, critical = action
        burst = 1 if (critical==1 and fidelity==2) else 0

        # Aliases / bins
        E = dT; Q = bufT; A = 1 if anomT else 0

        # Weights for the utilities
        lambda_I = E/2 + A
        lambda_T = E/2 + A
        lambda_R = (L + D)/4
        lambda_C = 1

        # Utilities
        need = min(1.0, E/2.0 + A)
        cap  = (fidelity + 1)/3.0
        dem  = min(1.0, (E + Q)/3.0)

        U_info = min(1.0, cap * dem)
        U_time = (1.0 / (1.0 + interval/5.0 + D/2.0)) * need

        # ---moderate ΔT + good battery + clean link → gently prefer shorter intervals
        if E == 1 and B == 2 and A == 0 and L == 0 and D == 0:
            U_time *= 1.20   # +20% pull toward shorter intervals

        # Reliability utility (sharper mismatch on bad links)
        g = 2 if (L==2 or D==2) else (1 if (L==1 or D==1) else 0)
        m = abs(qos - g)
        link_sev = 0.5 * (L/2.0 + D/2.0)  # ∈ [0,1]
        U_rel = 1.0 - (m**2) * (0.6 + 0.8*link_sev)

        # Critical correctness utility
        severity = min(2.0, 1.0 + 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
        U_crit = severity if (critical == true_crit) else -severity

        # Costs
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

        # Idle bonus and safety penalty
        R_idle = self.rho if (E==0 and Q==0 and B==2 and qos==0 and interval==60 and fidelity==0 and critical==0) else 0.0
        eventy = 1 if (E==2 or Q==2 or A==1) else 0
        lazy   = 1 if (fidelity==0 or interval==60) else 0
        safety_pen = self.kappa if (eventy and lazy) else 0.0

        # --- accumulate fidelity preference adjustment for true criticals
        crit_fidelity_adj = 0.0
        if true_crit == 1 and critical == 1:
            target_f = 2 if (E == 2 or Q >= 1 or A == 1) else 1
            sev_f = min(1.0, 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
            scale = 1.0 if B >= 1 else 0.5

            if fidelity < target_f:
                crit_fidelity_adj -= scale * (0.15 + 0.25 * sev_f)   # <-- use sev_f
            elif fidelity == 2:
                crit_fidelity_adj += scale * (0.10 + 0.18 * sev_f)   # <-- use sev_f


        # Base reward
        reward = ((self.w_I*lambda_I)*U_info +
                  (self.w_T*lambda_T)*U_time +
                  (self.w_R*lambda_R)*U_rel  +
                  (self.w_C*lambda_C)*U_crit
                  - mu_E*C_energy - mu_X*C_net - mu_B*C_backlog
                  + R_idle - safety_pen)

        # Add the fidelity adjustment
        reward += crit_fidelity_adj

        # Severity-scaled shaping for critical correctness
        sev = min(1.0, 0.5*E/2.0 + 0.5*Q/2.0 + 0.6*A)
        bonus, penalty = (0.5 + 0.9*sev), (1.1 + 1.1*sev)

        if true_crit == 0 and critical == 1:
            reward -= 0.25 + 0.35*sev
        if true_crit == 1 and critical == 1:
            reward += bonus
        elif true_crit == 1 and critical == 0:
            reward -= penalty

        if eventy and (interval == 5 or burst):
            reward += 0.2

        return float(np.clip(reward / 4.0, -1.0, 1.0))



# ----- Noisy + Dueling Network -----
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
        mu_range = 1.0 / np.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / np.sqrt(self.in_f))
        self.bias_sigma.data.fill_(self.sigma0 / np.sqrt(self.out_f))
    def reset_noise(self):
        eps_in  = torch.randn(self.in_f, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_f, device=self.weight_mu.device)
        f = lambda x: x.sign().mul_(x.abs().sqrt_())
        eps_in, eps_out = f(eps_in), f(eps_out)
        self.weight_eps.copy_(eps_out.ger(eps_in)); self.bias_eps.copy_(eps_out)
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
        self.c_head = nn.Linear(64, 1)
    def reset_noise(self): self.adv.reset_noise(); self.val.reset_noise()
    def forward(self, x):
        h = F.relu(self.fc1(x)); h = F.relu(self.fc2(h))
        A = self.adv(h); V = self.val(h)
        q = V + A - A.mean(dim=1, keepdim=True)
        c = self.c_head(h)
        return q, c

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):  # α increased for positives
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = (self.alpha*targets + (1-self.alpha)*(1-targets)) * (1-pt)**self.gamma * bce
        return loss.mean()


# ----- Agent -----
class DQNAgent:
    def __init__(self, state_size, action_size, actions):
        self.actions = actions; self.action_size = len(actions)
        self.state_size = state_size

        self.memory = ReplayBuffer(20000)
        self.gamma = 0.99; self.n_step = 3; self.gamma_n = self.gamma ** self.n_step
        self._nstep_buf = deque(maxlen=self.n_step)

        self.epsilon = 1.0; self.epsilon_min = 0.05; self.epsilon_decay = 0.99997
        self.batch_size = 128; self.tau = 0.01
        self.coverage_bonus_beta = 1.0
        self.beta0, self.betaT = 0.4, 1.0
        self.total_steps = 0; self.max_total_steps = 1
        self.warmup_steps = 4000

        self.policy_net = DuelingNoisyDQN(state_size, action_size).to(device)
        self.target_net = DuelingNoisyDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-5)

        # bias-init crit head to prior prevalence (e.g., 15%)
        p0 = 0.15
        b = float(np.log(p0/(1-p0)))
        with torch.no_grad():
            self.policy_net.c_head.bias.fill_(b)
            self.target_net.c_head.bias.fill_(b)

        self.q_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.crit_loss_fn = FocalLoss(alpha=0.9, gamma=2.0)
        self.reward_calculator = AdaptiveRewardCalculator()

        # --- metrics ---
        self.metrics = {'episode_rewards': [], 'loss_values': [], 'energy_consumption': [],
                        'avg_latency': [], 'on_time_rate': [], 'drop_rate': [], 'late_rate': [],
                        'td_error':[], 'convergence_data': [], 'classification_accuracy': [],
                        'crit_precision':[], 'crit_recall':[], 'crit_f1':[], 'crit_prevalence':[],
                        'chosen_threshold':[], 'val_precision':[],'val_cm': [], 'val_recall':[], 'val_f2':[],
                        'tuning_log': []}

        self.num_states = (3**8) * (2**1)
        self.state_visits = defaultdict(int); self.state_action_visits = defaultdict(int)

        # ---- Gate & threshold & sampling (will be tuned by guardrails) ----
        self.gate_warmup      = 800
        self.gate_recall_req  = 0.60
        self.gate_base        = 0.35   # will be slightly reduced if recall stalls
        self.gate_ramp_steps  = 8000
        self.gate_min, self.gate_max = 0.20, 0.90
        self.crit_force_eps0, self.crit_force_epsT = 0.30, 0.05
        self.crit_thresh0, self.crit_threshT = 0.25, 0.45  # schedule
        self.ema_recall = 0.0
        self.adaptive_thr = None  # set by F2 sweep
        self.gate_hi, self.gate_lo = 0.60, 0.40
        self.learn_calls = 0; self.target_update_every = 16

        # threshold sweep window (may be narrowed by guardrails if precision collapses)
        self.thr_min, self.thr_max, self.thr_steps = 0.25, 0.45, 9

        # replay positive fraction (may be raised by guardrails)
        self.pos_frac = 0.65

        # small fixed validation set for threshold sweep
        self.val_states = self._build_val_set(n=200, seed=2024)

        # internal guardrail flags
        self._recall_stall_hits = 0
        self._precision_bad_hits = 0

    def _build_val_set(self, n=200, seed=2024):
        rng = np.random.RandomState(seed)
        vals = []
        for _ in range(n):
            dT   = int(rng.randint(0,3))
            bufT = int(rng.randint(0,3))
            B    = int(rng.randint(0,3))
            L    = int(rng.randint(0,3))
            D    = int(rng.randint(0,3))
            Icur = int(rng.randint(0,3))
            Qcur = int(rng.randint(0,3))
            Fcur = int(rng.randint(0,3))
            anom = int(rng.randint(0,2))
            vals.append((dT,bufT,B,L,D,Icur,Qcur,Fcur,anom))
        return vals

    def set_training_horizon(self, total_steps): self.max_total_steps = max(total_steps, 1)

    # 8 ternaries + 1 binary (anomT)
    def one_hot_encode(self, state):
        encoded = []
        for v in state[:8]:
            v = int(v); v = v if v in (0,1,2) else 0
            encoded.extend(np.eye(3, dtype=np.float32)[v])
        encoded.append(float(state[8]))  # anomT
        return np.array(encoded, dtype=np.float32)  # size 25

    def get_state_tensor(self, state): return torch.from_numpy(self.one_hot_encode(state)).unsqueeze(0).to(device)
    def index_to_action(self, idx): return self.actions[idx]

    def _progress(self):
        return 0.0 if self.max_total_steps <= 0 else min(1.0, self.total_steps / float(self.max_total_steps))

    def _severity(self, state):
        dT, bufT, B, L, D, *_rest, anom = map(int, state)
        # temperature/backlog/anomaly + network health
        s = 0.35*(dT==2) + 0.35*(bufT==2) + 0.5*anom + 0.2*(dT==1) + 0.2*(bufT==1)
        s += 0.25*(L>=1 or D>=1) + 0.35*(L==2 or D==2)         # NEW: network contribution
        s += 0.15*(B==0 and (L>=1 or D>=1))                    # low battery + bad net boosts risk
        return float(min(1.0, s))


    def _gate_enabled(self):
        return (self.total_steps >= self.gate_warmup) and (self.ema_recall >= self.gate_recall_req)

    def _gate_strength(self):
        if not self._gate_enabled(): return 0.0
        return min(1.0, (self.total_steps - self.gate_warmup) / float(self.gate_ramp_steps))

    def _crit_force_prob(self, state):
        base = self.crit_force_epsT + (self.crit_force_eps0 - self.crit_force_epsT) * (1.0 - self._progress())
        return min(0.7, base + 0.5*self._severity(state))

    def _scheduled_threshold(self):
        return self.crit_thresh0 + (self.crit_threshT - self.crit_thresh0) * self._progress()

    def _current_threshold(self):
        return float(self.adaptive_thr) if (self.adaptive_thr is not None) else self._scheduled_threshold()

    def _gate_band(self, sev):
        thr = self._current_threshold()
        # severity shrinks the margin; keep at least ±0.05 around thr
        margin = max(0.05, 0.18 * (1.0 - sev))
        hi = min(0.95, thr + margin)
        lo = max(0.05, thr - margin)
        return hi, lo


    def _logic_prior_penalty(self, action, state, p_crit):
        dT, bufT, B, L, D, _, _, _, anom = map(int, state)
        sev = self._severity(state)
        net_bad = max(L, D)
        pred_c = int(p_crit >= self._current_threshold())

        qos, interval, fidelity, _ = action
        pen = 0.0

        if pred_c == 1:
            if interval == 60: pen += 0.6 + 0.6*sev
            elif interval == 30: pen += 0.3 + 0.3*sev
            if qos == 0: pen += 0.5 + 0.3*net_bad
            if fidelity == 0 and (sev > 0.5 or anom): pen += 0.3
        else:
            if interval == 5 and sev < 0.6: pen += 0.4*(0.6 - sev)
            if qos >= 1 and net_bad == 0: pen += 0.25*qos
            if fidelity == 2 and sev < 0.6: pen += 0.2

        if B == 0:
            pen += 0.30*(qos == 2) + 0.25*(interval == 5) + 0.20*(fidelity == 2)

        return float(min(1.0, pen))
    def _masked_actions(self, state):
        dT, bufT, B, L, D, *_ , anom = map(int, state)
        mask = []
        for i, a in enumerate(self.actions):
            qos, interval, fidelity, c = a

            # (A) existing danger masks
            if (bufT == 2 or dT == 2 or anom == 1) and interval == 60:
                continue
            if max(L, D) == 2 and qos == 0:
                continue

            # (B) NEW: moderate ΔT with plenty of battery + clean link
            # Nudge the policy away from very slow reporting when conditions allow it.
            good_net = (L == 0 and D == 0)
            if dT == 1 and B == 2 and anom == 0 and good_net and interval == 60 and c == 0:
                # forbid 60s; 5s or 30s remain
                continue

            mask.append(i)
        return mask

    def _apply_mask(self, indices, state):
        allowed = set(self._masked_actions(state))
        if not allowed:
            return list(indices)  # fallback: no mask
        out = [i for i in indices if i in allowed]
        return out if out else list(allowed)  # fallback: any allowed action


    def _argmax_soft_gate(self, q, p_crit: float, state=None):
        gs = self._gate_strength()
        if gs <= 0.0:
            # even here, respect mask
            cand = self._apply_mask(range(self.action_size), state)
            return int(cand[int(np.argmax([q[i] for i in cand]))])

        sev = self._severity(state) if state is not None else 0.0
        hi, lo = self._gate_band(sev)

        if p_crit >= hi:
            cand = [i for i, a in enumerate(self.actions) if a[3] == 1]
            cand = self._apply_mask(cand, state)
            return cand[int(np.argmax([q[i] for i in cand]))]

        if p_crit <= lo:
            cand = [i for i, a in enumerate(self.actions) if a[3] == 0]
            cand = self._apply_mask(cand, state)
            return cand[int(np.argmax([q[i] for i in cand]))]

        # Mixed band: soften scores, then hard-block masked ones
        q = q.copy()
        rng  = float(q.max() - q.min() + 1e-6)
        base = self.gate_base * rng * gs

        for i, a in enumerate(self.actions):
            mismatch_prob = (p_crit if a[3] == 0 else (1.0 - p_crit))
            logic = self._logic_prior_penalty(a, state, p_crit)
            q[i] -= base * (0.7*mismatch_prob + 0.3*logic) * (0.6 + 0.8*sev)

        allowed = set(self._masked_actions(state))
        if allowed:
            for i in range(self.action_size):
                if i not in allowed:
                    q[i] = -1e9  # forbid

        return int(np.argmax(q))


    # ---- n-step storage helpers ----
    def store_transition(self, s, a_idx, r, ns, done, true_crit):
        self._nstep_buf.append((s, a_idx, r, ns, done))
        if len(self._nstep_buf) < self.n_step: return
        R, ns_n, d = 0.0, self._nstep_buf[-1][3], self._nstep_buf[-1][4]
        for k, (_, _, rk, _, dk) in enumerate(self._nstep_buf):
            R += (self.gamma ** k) * rk
            if dk: d = True; break
        s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
        seed = 1.0 + 4.0*true_crit + 4.0*int(self.actions[a0][3] != true_crit)
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

    def select_action(self, state, explore=True, training=True):
        if training and hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()

        if training:
            state_idx = 0
            for v in state[:8]:
                state_idx = state_idx * 3 + int(v)
            state_idx = state_idx * 2 + int(state[8])
            self.state_visits[state_idx] += 1
            counts = [self.state_action_visits.get((state_idx, a), 0) for a in range(self.action_size)]
            inv_counts = np.array([1.0 / (1 + c) for c in counts], dtype=np.float32)
        else:
            inv_counts = None; state_idx = None

        st = self.get_state_tensor(state)
        with torch.no_grad():
            q_values_t, c_logit_t = self.policy_net(st)
            q_values = q_values_t.cpu().numpy().flatten()

            logit = float(c_logit_t.item())
            T = getattr(self, "temp_scale", None)
            if T is not None and T > 0: logit = logit / float(T)

            p_crit = 1.0 / (1.0 + np.exp(-logit))

            if training and inv_counts is not None:
                q_values = q_values + self.coverage_bonus_beta * inv_counts

        sev = self._severity(state)
        anom = int(state[8])

        # Safety override
        if (sev >= 0.80 and p_crit >= 0.20) or anom == 1 or int(state[1]) == 2 or max(int(state[3]), int(state[4])) == 2:
            cand = [i for i, a in enumerate(self.actions) if a[3] == 1]
            cand = self._apply_mask(cand, state)
            action_idx = cand[int(np.argmax([q_values[i] for i in cand]))]
            if training and state_idx is not None:
                self.state_action_visits[(state_idx, action_idx)] += 1
            return action_idx

        if explore and training:
        # Forced critical exploration
            if random.random() < self._crit_force_prob(state):
                valid = [i for i, a in enumerate(self.actions) if a[3] == 1]
                valid = self._apply_mask(valid, state)
                action_idx = random.choice(valid)

            # ε-greedy
            elif random.random() < self.epsilon:
                if self._gate_enabled():
                    thr = self._current_threshold()
                    pred_crit = int(p_crit >= thr)
                    valid = [i for i, a in enumerate(self.actions) if a[3] == pred_crit]
                    valid = self._apply_mask(valid, state)
                    action_idx = random.choice(valid)
                else:
                    # Use inv_counts probabilities but zero out disallowed actions
                    if inv_counts is not None:
                        probs = inv_counts.copy()
                        allowed = set(self._masked_actions(state))
                        if allowed:
                            for i in range(self.action_size):
                                if i not in allowed:
                                    probs[i] = 0.0
                            s = probs.sum()
                            if s > 0:
                                probs = probs / s
                                action_idx = np.random.choice(self.action_size, p=probs)
                            else:
                                # fallback if mask nuked everything
                                valid = list(allowed) if allowed else list(range(self.action_size))
                                action_idx = random.choice(valid)
                        else:
                            # no mask; normal
                            probs = probs / (probs.sum() + 1e-12)
                            action_idx = np.random.choice(self.action_size, p=probs)
                    else:
                        valid = self._apply_mask(range(self.action_size), state)
                        action_idx = random.choice(valid)
            else:
                # Exploit path still goes through soft gate (already masked inside)
                action_idx = self._argmax_soft_gate(q_values, p_crit, state)
        else:
            # Evaluation/inference — masked inside soft gate
            action_idx = self._argmax_soft_gate(q_values, p_crit, state)

        ''' if explore and training:
            if random.random() < self._crit_force_prob(state):
                valid = [i for i, a in enumerate(self.actions) if a[3] == 1]
                valid = self._apply_mask(valid, state)
                action_idx = random.choice(valid)
            elif random.random() < self.epsilon:
                probs = None
                if self._gate_enabled():
                    thr = self._current_threshold()
                    pred_crit = int(p_crit >= thr)
                    valid = [i for i, a in enumerate(self.actions) if a[3] == pred_crit]
                    valid = self._apply_mask(valid, state)
                    action_idx = random.choice(valid)
                else:
                    if inv_counts is not None:
                        probs = inv_counts / (inv_counts.sum() + 1e-12)
                    action_idx = np.random.choice(self.action_size, p=probs) if probs is not None else random.randrange(self.action_size)
            else:
                action_idx = self._argmax_soft_gate(q_values, p_crit, state)
        else:
            action_idx = self._argmax_soft_gate(q_values, p_crit, state) '''

        if training:
            self.state_action_visits[(state_idx, action_idx)] += 1
        return action_idx

    def _crit_weight(self):
        return 0.7

    # ---- Validation sweep for F2 to set threshold ----
    @staticmethod
    def _f_beta(precision, recall, beta=2.0, eps=1e-9):
        b2 = beta*beta
        return (1+b2) * (precision*recall) / (b2*precision + recall + eps)

    # after each episode, sample 200 states from recent replay for the next sweep window
    def _refresh_val_set_from_replay(self, n=200):
        idx = np.flatnonzero(self.memory.filled)
        if idx.size:
            pick = np.random.choice(idx, size=min(n, idx.size), replace=False)
            self.val_states = [self.memory.buffer[i].state for i in pick]

    def _sweep_threshold(self):
        states = self.val_states
        with torch.no_grad():
            X = torch.from_numpy(np.stack([self.one_hot_encode(s) for s in states], axis=0)).to(device)
            _, logits = self.policy_net(X)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        labels = np.array([compute_true_criticality(s) for s in states], dtype=np.int32)

        thrs = np.linspace(self.thr_min, self.thr_max, self.thr_steps)
        best_thr, best_f2, best_p, best_r = thrs[0], -1.0, 0.0, 0.0

        for t in thrs:
            preds = (probs >= t).astype(np.int32)
            tp = np.sum((preds==1) & (labels==1))
            fp = np.sum((preds==1) & (labels==0))
            fn = np.sum((preds==0) & (labels==1))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f2 = self._f_beta(prec, rec, beta=2.0)
            if f2 > best_f2:
                best_f2, best_thr, best_p, best_r = f2, float(t), float(prec), float(rec)

        self.adaptive_thr = best_thr
        self.metrics['chosen_threshold'].append(best_thr)
        self.metrics['val_precision'].append(best_p)
        self.metrics['val_recall'].append(best_r)
        self.metrics['val_f2'].append(best_f2)
        # NEW: confusion matrix
        preds = (probs >= best_thr).astype(np.int32)
        tp = int(np.sum((preds==1) & (labels==1)))
        fp = int(np.sum((preds==1) & (labels==0)))
        tn = int(np.sum((preds==0) & (labels==0)))
        fn = int(np.sum((preds==0) & (labels==1)))
        self.metrics['val_cm'].append((tp, fp, tn, fn))
        print(f"[Val sweep] thr={best_thr:.2f}  P={best_p:.2f} R={best_r:.2f} F2={best_f2:.2f}  CM: tp={tp} fp={fp} tn={tn} fn={fn}")

    # evaluation on probe states (unchanged)
    def _evaluate_policy(self, episode):
        # Refresh validation set from the most recent replay states
        self._refresh_val_set_from_replay(n=200)
        # Now sweep threshold on these on-policy states
        self._sweep_threshold()

        test_states = [
            (0,0,2,0,0,1,1,1,0),
            (1,0,2,0,0,1,0,0,0),
            (1,1,2,1,0,2,1,1,0),
            (2,0,2,0,0,0,2,2,0),
            (0,2,2,1,1,1,1,0,0),
            (1,1,0,2,2,1,2,1,0),
            (2,2,2,0,0,0,2,2,1),
            (1,1,1,1,1,1,1,1,1),
            (0,2,1,1,1,2,0,0,0),
            (2,0,1,0,1,0,2,2,1),
        ]
        action_counts = np.zeros((3, 3)); crit_counts = np.zeros(2); state_q_values = {}

        for s in test_states:
            st = self.get_state_tensor(s)
            with torch.no_grad():
                q_values_t, c_logit = self.policy_net(st)
                q_values = q_values_t.cpu().numpy().flatten()
                _ = float(torch.sigmoid(c_logit).item())
                state_q_values[s] = float(np.max(q_values))

            idx = self.select_action(s, explore=False, training=False)
            qos, interval, fidelity, critical = self.index_to_action(idx)
            interval_idx = [5,30,60].index(interval)
            action_counts[qos, interval_idx] += 1
            crit_counts[critical] += 1

        self.metrics['convergence_data'].append({
            'episode': episode,
            'action_dist': action_counts / len(test_states),
            'crit_dist': crit_counts / len(test_states),
            'state_q_values': state_q_values
        })

    def _get_ideal_action(self, state):
        max_r = -1e9; best = None; true_crit = compute_true_criticality(state)
        for a in self.actions:
            r = self.reward_calculator.calculate_reward(state, a, 0.0, true_crit)
            if r > max_r: max_r, best = r, a
        return best

    def analyze_policy(self):
        print("\nDouble DQN Policy Analysis (T-only):")
        test_states = [
            (0,0,2,0,0,1,1,1,0),
            (1,0,2,0,0,1,0,0,0),
            (1,1,2,1,0,2,1,1,0),
            (2,0,2,0,0,0,2,2,0),
            (0,2,2,1,1,1,1,0,0),
            (1,1,0,2,2,1,2,1,0),
            (2,2,2,0,0,0,2,2,1),
            (1,1,1,1,1,1,1,1,1),
            (0,2,1,1,1,2,0,0,0),
            (2,0,1,0,1,0,2,2,1),
        ]
        for state in test_states:
            st = self.get_state_tensor(state)
            with torch.no_grad():
                q_values_t, c_logit = self.policy_net(st)
                q_values = q_values_t.cpu().numpy().flatten()
                p = float(torch.sigmoid(c_logit).item())
            learned_idx = self.select_action(state, explore=False, training=False)

            learned_action = self.index_to_action(learned_idx)
            ideal_action = self._get_ideal_action(state)
            true_crit = compute_true_criticality(state)
            reward = self.reward_calculator.calculate_reward(state, learned_action, 0.0, true_crit)

            print(f"\nState (ΔT={state[0]}, buf_T={state[1]}, B={state[2]}, L={state[3]}, D={state[4]}, anomT={state[8]}):")
            print(f"  Learned action: QoS={learned_action[0]}, Interval={learned_action[1]}s, Fidelity={learned_action[2]}, Critical={learned_action[3]}")
            print(f"  Ideal action:   QoS={ideal_action[0]}, Interval={ideal_action[1]}s, Fidelity={ideal_action[2]}, Critical={ideal_action[3]}")
            print(f"  Reward: {reward:.2f}, Critical match: {'✓' if learned_action[3] == true_crit else '✗'} (True: {true_crit})")

            top_actions = np.argsort(q_values)[-3:][::-1]
            for i, act_idx in enumerate(top_actions):
                q, tau, f, c = self.index_to_action(act_idx)
                ideal = " (IDEAL)" if (q, tau, f, c) == ideal_action else ""
                print(f"  Top {i+1}: QoS={q}, τ={tau}s, Fidelity={f}, Critical={c}{ideal} (Q={q_values[act_idx]:.2f})")

    def learn(self):
        if len(self.memory) < self.batch_size: return None

        frac = 0.0 if self.max_total_steps <= 0 else min(1.0, self.total_steps / float(self.max_total_steps))
        beta = self.beta0 + (self.betaT - self.beta0) * frac

        # use tunable pos_frac (guardrails may bump this)
        batch, indices, weights = self.memory.sample_stratified(self.batch_size, beta=beta, pos_frac=self.pos_frac)

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
            target_q = reward_batch + (1 - done_batch) * self.gamma_n * next_q

        td_per_elem = self.q_loss_fn(current_q, target_q)
        q_loss = (td_per_elem * weights_t).mean()

        # --- crit head on CURRENT state ---
        crit_labels = torch.FloatTensor([compute_true_criticality(s) for s in batch.state]).unsqueeze(1).to(device)
        crit_loss = self.crit_loss_fn(c_logit, crit_labels)

        # metrics for classifier (on-batch, current-state)
        with torch.no_grad():
            preds = (torch.sigmoid(c_logit) >= self._current_threshold()).float()
            tp = ((preds == 1) & (crit_labels == 1)).sum().item()
            fp = ((preds == 1) & (crit_labels == 0)).sum().item()
            fn = ((preds == 0) & (crit_labels == 1)).sum().item()
            precision = tp / (tp + fp + 1e-6); recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            self.metrics['crit_precision'].append(precision)
            self.metrics['crit_recall'].append(recall)
            self.metrics['crit_f1'].append(f1)
            self.metrics['crit_prevalence'].append(float(crit_labels.mean().item()))
            self.ema_recall = 0.95*self.ema_recall + 0.05*recall

        loss = q_loss + 1.0 * crit_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if hasattr(self.policy_net, "reset_noise"): self.policy_net.reset_noise()

        with torch.no_grad():
            td_error = (current_q - target_q).abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_error)
        if (self.total_steps % 10) == 0: self.metrics['td_error'].append(float(td_error.mean()))

        self.learn_calls += 1
        if (self.learn_calls % self.target_update_every) == 0:
            with torch.no_grad():
                tau = self.tau
                for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * pp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1
        return float(loss.item())

    def plot_metrics(self):
        plt.figure(figsize=(18, 44))
        window = max(1, len(self.metrics['episode_rewards']) // 20)

        plt.subplot(6,3,1)
        rewards = self.metrics['episode_rewards']
        if rewards:
            smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(rewards, alpha=0.3); plt.plot(range(window-1,len(rewards)), smooth)
        plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title('DDQN - Learning Curve'); plt.grid(True)

        plt.subplot(6,3,2)
        acc = self.metrics['classification_accuracy']
        if acc:
            smooth = np.convolve(acc, np.ones(window)/window, mode='valid')
            plt.plot(acc, alpha=0.3); plt.plot(range(window-1,len(acc)), smooth)
        plt.xlabel('Episode'); plt.ylabel('Accuracy'); plt.title('Critical Classification Accuracy (proxy)'); plt.grid(True)

        plt.subplot(6,3,3)
        energy = self.metrics['energy_consumption']
        if energy:
            smooth = np.convolve(energy, np.ones(window)/window, mode='valid')
            plt.plot(energy, alpha=0.3); plt.plot(range(window-1,len(energy)), smooth)
        plt.xlabel('Episode'); plt.ylabel('Energy'); plt.title('Energy Consumption per Episode'); plt.grid(True)

        plt.subplot(6,3,4)
        on_time = self.metrics['on_time_rate']; dropped = self.metrics['drop_rate']
        if on_time and dropped:
            s1 = np.convolve(on_time, np.ones(window)/window, mode='valid')
            s2 = np.convolve(dropped, np.ones(window)/window, mode='valid')
            plt.plot(on_time, alpha=0.3, color='green', label='On-time')
            plt.plot(range(window-1,len(on_time)), s1, color='green')
            plt.plot(dropped, alpha=0.3, color='red', label='Dropped')
            plt.plot(range(window-1,len(dropped)), s2, color='red')
        plt.xlabel('Episode'); plt.ylabel('Rate'); plt.title('Packet Reliability Metrics'); plt.legend(); plt.grid(True)

        plt.subplot(6,3,5)
        latency = self.metrics['avg_latency']
        if latency:
            s = np.convolve(latency, np.ones(window)/window, mode='valid')
            plt.plot(latency, alpha=0.3); plt.plot(range(window-1,len(latency)), s)
        plt.xlabel('Episode'); plt.ylabel('Latency (ms)'); plt.title('Average Delivery Latency'); plt.grid(True)

        plt.subplot(6,3,6)
        if self.metrics['convergence_data']:
            final_actions = self.metrics['convergence_data'][-1]['action_dist']
            sns.heatmap(final_actions, annot=True, fmt='.2f',
                        xticklabels=['5s','30s','60s'],
                        yticklabels=['QoS0','QoS1','QoS2'], cmap='YlGnBu')
            plt.title('Final Action Distribution (QoS × Interval)'); plt.xlabel('Interval'); plt.ylabel('QoS')

        plt.subplot(6,3,7)
        if self.metrics['convergence_data']:
            crit_data = [x['crit_dist'][1] for x in self.metrics['convergence_data']]
            episodes = [x['episode'] for x in self.metrics['convergence_data']]
            plt.plot(episodes, crit_data)
        plt.xlabel('Episode'); plt.ylabel('Critical Flag Rate');
        plt.axhspan(0.35, 0.45, color='gray', alpha=0.10, label='Target band')
        plt.title('Critical Transmission Rate Over Time'); plt.grid(True)

        plt.subplot(6,3,8)
        losses = self.metrics['loss_values']
        if losses:
            wl = max(1, len(losses)//20)
            smooth = np.convolve(losses, np.ones(wl)/wl, mode='valid')
            plt.plot(losses, alpha=0.3); plt.plot(range(wl-1,len(losses)), smooth)
        plt.xlabel('Training Step'); plt.ylabel('Loss'); plt.title('DDQN + Crit Head Training Loss'); plt.grid(True)

        plt.subplot(6,3,9)
        energy = self.metrics['energy_consumption']; rewards = self.metrics['episode_rewards']
        if energy and rewards:
            plt.scatter(energy, rewards, alpha=0.5)
        plt.xlabel('Energy'); plt.ylabel('Episode Reward'); plt.title('Reward vs Energy'); plt.grid(True)

        plt.subplot(6,3,10)
        if self.metrics['convergence_data']:
            ts = list(self.metrics['convergence_data'][-1]['state_q_values'].keys())
            for s in ts:
                episodes = [d['episode'] for d in self.metrics['convergence_data']]
                qv = [d['state_q_values'][s] for d in self.metrics['convergence_data']]
                plt.plot(episodes, qv, label=str(s))
        plt.xlabel('Episode'); plt.ylabel('Max Q'); plt.title('Q-value Convergence (probe states)'); plt.legend(fontsize=7); plt.grid(True)


        plt.subplot(6,3,11)
        P, R, F = self.metrics['crit_precision'], self.metrics['crit_recall'], self.metrics['crit_f1']
        if P:
            def sm(x):
                w = max(1, len(x)//20)
                if len(x) < w: return x
                return np.convolve(x, np.ones(w)/w, mode='valid')
            p, r, f = sm(P), sm(R), sm(F)
            xp = range(len(p)); xr = range(len(r)); xf = range(len(f))
            plt.plot(xp, p, label='Precision'); plt.plot(xr, r, label='Recall'); plt.plot(xf, f, label='F1'); plt.legend()
        plt.xlabel('Training Step'); plt.ylabel('Score'); plt.title('Critical Classifier: P/R/F1'); plt.grid(True)


        plt.subplot(6,3,12)
        if self.metrics['crit_prevalence']:
            plt.plot(self.metrics['crit_prevalence'])
        plt.xlabel('Training Step'); plt.ylabel('Prevalence'); plt.title('Critical State Prevalence in Batches'); plt.grid(True)

        plt.subplot(6,3,13)
        if self.metrics['td_error']:
            td = self.metrics['td_error']; wtd = max(1, len(td)//20)
            smooth = np.convolve(td, np.ones(wtd)/wtd, mode='valid')
            plt.plot(td, alpha=0.3); plt.plot(range(wtd-1,len(td)), smooth)
        plt.xlabel('Training Step'); plt.ylabel('TD Error'); plt.title('TD Error (abs)'); plt.grid(True)

        # after subplot(6,3,11) add:
        plt.subplot(6,3,14)
        vp, vr, vf = self.metrics.get('val_precision', []), self.metrics.get('val_recall', []), self.metrics.get('val_f2', [])
        if vp:
            x = range(len(vp))
            plt.plot(x, vp, label='Val Precision')
            plt.plot(x, vr, label='Val Recall')
            plt.plot(x, vf, label='Val F2')
            plt.legend()
        plt.xlabel('Eval #'); plt.title('Held-out P/R/F2 over sweeps'); plt.grid(True)


        plt.subplot(6,3,14)
        vp = self.metrics.get('val_precision', [])
        vr = self.metrics.get('val_recall', [])
        vf = self.metrics.get('val_f2', [])
        if len(vp):
            x = range(len(vp))
            plt.plot(x, vp, label='Val Precision')
            plt.plot(x, vr, label='Val Recall')
            plt.plot(x, vf, label='Val F2')
            plt.ylim(0.0, 1.0)          # make tiny values visible
            plt.legend()
        plt.xlabel('Eval #'); plt.title('Held-out P/R/F2 over sweeps'); plt.grid(True)


        plt.tight_layout(); plt.show()


# ======== LIGHT TUNING GUARDRAILS ========
class TuningGuardrails:
    @staticmethod
    def estimate_prevalence(env_cfg, steps=3000):
        """Rough natural positive rate using a neutral policy."""
        env = RealisticEnvSimulator(**env_cfg)
        state = (0,0,2,0,0,1,1,1,0)
        pos = n = 0
        neutral = (0, 30, 1, 0)
        for _ in range(steps):
            next_state, *_ = env.simulate_environment(state, neutral)
            true_crit = compute_true_criticality(next_state)
            pos += true_crit; n += 1
            state = next_state
        return pos / max(1, n)

    @staticmethod
    def auto_calibrate_environment(target_low=0.10, target_high=0.25, max_passes=3):
        """
        Adjust CRIT_THRESH and env knobs (anom_threshold, a_scale) to
        land the natural positives in [target_low, target_high].
        """
        global CRIT_THRESH
        env_cfg = {'anom_threshold': 2.5, 'a_scale': 1.0}

        for _ in range(max_passes):
            prev = TuningGuardrails.estimate_prevalence(env_cfg, steps=1500)
            print(f"[Calibrator] Estimated prevalence={prev:.3f}  (CRIT_THRESH={CRIT_THRESH}, "
                  f"anom_thr={env_cfg['anom_threshold']:.2f}, a_scale={env_cfg['a_scale']:.2f})")
            if target_low <= prev <= target_high:
                print("[Calibrator] OK — within target range.")
                break
            if prev < target_low:
                # too rare → make positive easier
                if CRIT_THRESH > 3:
                    CRIT_THRESH -= 1
                    print(f"[Calibrator] Lowering CRIT_THRESH → {CRIT_THRESH}")
                else:
                    env_cfg['anom_threshold'] = max(1.5, env_cfg['anom_threshold'] * 0.90)
                    env_cfg['a_scale'] = min(2.0, env_cfg['a_scale'] * 1.15)
                    print(f"[Calibrator] anom_threshold↓ → {env_cfg['anom_threshold']:.2f}, a_scale↑ → {env_cfg['a_scale']:.2f}")
            else:
                # too common → make positive harder
                if CRIT_THRESH < 6:
                    CRIT_THRESH += 1
                    print(f"[Calibrator] Raising CRIT_THRESH → {CRIT_THRESH}")
                else:
                    env_cfg['anom_threshold'] = min(4.0, env_cfg['anom_threshold'] * 1.10)
                    env_cfg['a_scale'] = max(0.6, env_cfg['a_scale'] * 0.90)
                    print(f"[Calibrator] anom_threshold↑ → {env_cfg['anom_threshold']:.2f}, a_scale↓ → {env_cfg['a_scale']:.2f}")
        return env_cfg

    @staticmethod
    def adjust_after_eval(agent: DQNAgent):
        """
        Lightweight reactive tweaks based on last eval:
         - If recall < 0.70 for two evals → increase pos_frac to 0.70 (cap) and reduce gate_base by 10%.
         - If precision < 0.55 → narrow F2 sweep to 0.30–0.40 until it recovers.
        """
        tp, fp, tn, fn = agent.metrics['val_cm'][-1] if agent.metrics.get('val_cm') else (0,0,0,0)
        fp_rate = fp / (fp + tn + 1e-9)
        last_r = agent.metrics['val_recall'][-1] if agent.metrics['val_recall'] else None
        last_p = agent.metrics['val_precision'][-1] if agent.metrics['val_precision'] else None

        # Recall stall guard
        if last_r is not None and last_r < 0.70:
            agent._recall_stall_hits += 1
        else:
            agent._recall_stall_hits = 0

        if agent._recall_stall_hits >= 2:
            old_pf, old_gb = agent.pos_frac, agent.gate_base
            agent.pos_frac = min(0.70, agent.pos_frac + 0.05)
            agent.gate_base = max(0.22, agent.gate_base * 0.90)   # one extra 10% relaxation
            agent._recall_stall_hits = 0
            agent.metrics['tuning_log'].append(
                f"[Guardrails] Recall stalled → pos_frac {old_pf:.2f}->{agent.pos_frac:.2f}, gate_base {old_gb:.2f}->{agent.gate_base:.2f}"
            )
            print(agent.metrics['tuning_log'][-1])

        # Precision collapse guard
        if last_p is not None and last_p < 0.55:
            agent._precision_bad_hits += 1
        else:
            agent._precision_bad_hits = 0


        if agent._precision_bad_hits >= 1:
            if not (abs(agent.thr_min-0.30) < 1e-6 and abs(agent.thr_max-0.40) < 1e-6):
                old = (agent.thr_min, agent.thr_max)
                agent.thr_min, agent.thr_max, agent.thr_steps = 0.30, 0.40, 9
                agent.metrics['tuning_log'].append(
                    f"[Guardrails] Precision low → F2 sweep window {old[0]:.2f}-{old[1]:.2f} -> 0.30-0.40"
                )
                print(agent.metrics['tuning_log'][-1])

        if last_r is not None and last_p is not None and last_r > 0.90 and (last_p < 0.60 or fp_rate > 0.20):
            old = (agent.thr_min, agent.thr_max)
            # shift center up by +0.05; keep width ~0.10; clamp
            center = min(0.65, (old[0] + old[1]) / 2.0 + 0.05)
            agent.thr_min = max(0.30, center - 0.05)
            agent.thr_max = min(0.70, center + 0.05)
            agent.thr_steps = 9
            agent.metrics['tuning_log'].append(
                f"[Guardrails] High R/low P → sweep {old[0]:.2f}-{old[1]:.2f} → {agent.thr_min:.2f}-{agent.thr_max:.2f}"
            )
            print(agent.metrics['tuning_log'][-1])
        # Optional: anti-spam monitor (keeps gate honest)
        if agent.metrics['convergence_data']:
            crit_rate = agent.metrics['convergence_data'][-1]['crit_dist'][1]
            if crit_rate > 0.8:
                # too many 1's → softly raise threshold window upper bound next sweep
                agent.thr_max = max(agent.thr_max, 0.45)
            if crit_rate < 0.2:
                # too few 1's → softly lower window lower bound next sweep
                agent.thr_min = min(agent.thr_min, 0.25)


# ----- Training loop -----
def train_dqn(agent, episodes=700, max_steps=250, env_cfg=None):
    env_cfg = env_cfg or {}
    agent.set_training_horizon(total_steps=episodes * max_steps)

    for episode in tqdm(range(episodes), desc="Training"):
        env = RealisticEnvSimulator(**env_cfg)
        total_reward = total_energy = total_latency = 0.0
        steps = correct_critical = 0
        pkt = {'on_time':0, 'late':0, 'dropped':0}

        # initial state (dT, bufT, B, L, D, I_cur, Q_cur, F_cur, anomT)
        state = (0,0,2,0,0,1,1,1,0)

        for _ in range(max_steps):
            a_idx = agent.select_action(state)
            action = agent.index_to_action(a_idx)

            next_state, pkt_status, true_crit, energy_used, latency = env.simulate_environment(state, action)
            reward = agent.reward_calculator.calculate_reward(state, action, energy_used, true_crit)

            done = env.battery_mAh <= 0
            agent.store_transition(state, a_idx, reward, next_state, done, true_crit)

            loss = agent.learn()
            if loss is not None: agent.metrics['loss_values'].append(loss)

            total_reward += reward; total_energy += energy_used; total_latency += latency
            pkt[pkt_status] += 1; correct_critical += int(action[3] == true_crit); steps += 1
            state = next_state
            if done: break

        agent.flush_nstep()

        # episode metrics
        agent.metrics['episode_rewards'].append(total_reward)
        agent.metrics['energy_consumption'].append(total_energy)
        agent.metrics['avg_latency'].append(total_latency/steps if steps else 0.0)
        agent.metrics['on_time_rate'].append(pkt['on_time']/steps if steps else 0.0)
        agent.metrics['drop_rate'].append(pkt['dropped']/steps if steps else 0.0)
        agent.metrics['late_rate'].append(pkt['late']/steps if steps else 0.0)
        agent.metrics['classification_accuracy'].append(correct_critical/steps if steps else 0.0)

        if episode % 30 == 0 or episode == episodes - 1:
            agent._evaluate_policy(episode)
            # apply reactive guardrails AFTER we gather eval metrics
            TuningGuardrails.adjust_after_eval(agent)
            print(f"Episode {episode}  Recall_EMA={agent.ema_recall:.2f}  Thr={agent._current_threshold():.2f}  "
                  f"ε={agent.epsilon:.3f}  Gate={agent._gate_enabled()}  pos_frac={agent.pos_frac:.2f}  "
                  f"sweep=[{agent.thr_min:.2f},{agent.thr_max:.2f}]")

    return agent


def run_experiment():
    global intervals
    intervals = [5, 30, 60]

    actions = [(q, t, f, c)
               for q in [0, 1, 2]
               for t in [5, 30, 60]
               for f in [0, 1, 2]
               for c in [0, 1]]

    state_size = 25  # 8*3 one-hots + 1 binary
    action_size = len(actions)

    # --- Step 0: auto-calibrate environment prevalence (10–25%) ---
    env_cfg = TuningGuardrails.auto_calibrate_environment(target_low=0.10, target_high=0.25, max_passes=3)

    agent = DQNAgent(state_size, action_size, actions)
    agent = train_dqn(agent, episodes=700, max_steps=250, env_cfg=env_cfg)
    agent.plot_metrics()
    agent.analyze_policy()

    total_pairs_encountered = len(agent.state_action_visits)
    unique_states_visited = len(agent.state_visits)
    print("\nCoverage quick view:",
          f"pairs_seen={total_pairs_encountered}",
          f"unique_states={unique_states_visited}")
    if agent.metrics['tuning_log']:
        print("\nTuning guardrails applied:")
        for line in agent.metrics['tuning_log']:
            print("  ", line)

if __name__ == "__main__":
    run_experiment()
