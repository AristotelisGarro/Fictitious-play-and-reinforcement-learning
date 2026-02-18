"""
Repeated Prisoner's Dilemma - Fictitious Play & Reinforcement Learning
=======================================================================
Supports:
  - Fictitious Play (FP) agents
  - Independent Q-Learning (IQL) agents
  - MiniMax Q-Learning agents
  - Matchups: RL vs RL, FP vs FP, RL vs FP
  - ε-greedy exploration strategy
  - Nash Equilibrium distance tracking
  - Configurable hyperparameters & reporting
  - Plots: cumulative rewards, cooperation rates, Nash distance
"""

import numpy as np
import random
import argparse
import json
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ─────────────────────────────────────────────
# Payoff Matrix  (row=Agent0, col=Agent1)
# Action: 0 = Cooperate (C), 1 = Defect (D)
# ─────────────────────────────────────────────
cooperation_reward = 3
defection_reward = 5
PAYOFF = {
    (0, 0): (cooperation_reward, cooperation_reward),   # C,C → Reward
    (0, 1): (0, defection_reward),   # C,D → Sucker / Temptation
    (1, 0): (defection_reward, 0),   # D,C → Temptation / Sucker
    (1, 1): (1, 1),   # D,D → Punishment (Nash Equilibrium)
}
ACTIONS = [0, 1]
ACTION_NAMES = {0: "C", 1: "D"}

# Nash Equilibrium of the stage game: both players Defect (D,D)
# In mixed strategy NE: both play D with probability 1 (pure NE)
NASH_P0_DEFECT = 1.0   # Nash equilibrium prob of defecting for player 0
NASH_P1_DEFECT = 1.0   # Nash equilibrium prob of defecting for player 1

# Colour palette per matchup
MATCHUP_COLORS = {
    "FP vs FP":             ("#2196F3", "#64B5F6"),
    "IQL vs IQL":           ("#E53935", "#EF9A9A"),
    "MINIMAX vs MINIMAX":   ("#43A047", "#A5D6A7"),
    "IQL vs FP":            ("#FB8C00", "#FFCC80"),
    "MINIMAX vs FP":        ("#8E24AA", "#CE93D8"),
    "IQL vs MINIMAX":       ("#00897B", "#80CBC4"),
}


# ─────────────────────────────────────────────
# Config Dataclass
# ─────────────────────────────────────────────
@dataclass
class Config:
    n_rounds: int = 1000
    # RL hyperparameters
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    # FP hyperparameters
    fp_smoothing: float = 1.0
    # MiniMax-Q specific
    minimax_lr: float = 0.1
    minimax_gamma: float = 0.9
    # Reporting
    report_interval: int = 100
    plot_window: int = 50        # rolling window for charts        

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────
# Nash Distance Utilities
# ─────────────────────────────────────────────
def compute_nash_distance(p0_defect: float, p1_defect: float) -> float:
    """
    Euclidean distance from the (empirical) joint mixed strategy
    to the pure Nash Equilibrium (p_defect=1 for both players).

    dist = sqrt((p0_defect - 1)^2 + (p1_defect - 1)^2)
    """
    return float(np.sqrt((p0_defect - NASH_P0_DEFECT) ** 2 +
                          (p1_defect - NASH_P1_DEFECT) ** 2))


def rolling_defect_rates(round_log: List[Dict], window: int) -> Tuple[List, List, List]:
    """
    Compute per-round rolling defection rates for both agents
    and the resulting Nash distance series.
    """
    n = len(round_log)
    d0, d1, nd = [], [], []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = round_log[start: i + 1]
        p0d = sum(1 for x in chunk if x["a0"] == 1) / len(chunk)
        p1d = sum(1 for x in chunk if x["a1"] == 1) / len(chunk)
        d0.append(p0d)
        d1.append(p1d)
        nd.append(compute_nash_distance(p0d, p1d))
    return d0, d1, nd


# ─────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────
class Agent:
    def __init__(self, name: str, agent_id: int):
        self.name = name
        self.agent_id = agent_id
        self.total_reward = 0.0
        self.history: List[Tuple[int, int]] = []
        self.rewards_per_round: List[float] = []

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        self.total_reward += reward
        self.history.append((my_action, opp_action))
        self.rewards_per_round.append(reward)

    def reset(self):
        self.total_reward = 0.0
        self.history = []
        self.rewards_per_round = []

    def cooperation_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for a, _ in self.history if a == 0) / len(self.history)

    def cumulative_rewards(self) -> List[float]:
        cum = []
        s = 0.0
        for r in self.rewards_per_round:
            s += r
            cum.append(s)
        return cum

    def __repr__(self):
        return f"{self.name}(id={self.agent_id})"


# ─────────────────────────────────────────────
# Fictitious Play Agent
# ─────────────────────────────────────────────
class FictitiousPlayAgent(Agent):
    def __init__(self, agent_id: int, config: Config):
        super().__init__("FP", agent_id)
        self.config = config
        self.opp_counts = [config.fp_smoothing, config.fp_smoothing]

    def select_action(self) -> int:
        total = sum(self.opp_counts)
        p_coop = self.opp_counts[0] / total
        p_defect = self.opp_counts[1] / total
        ev_coop = p_coop * cooperation_reward + p_defect * 0
        ev_defect = p_coop * defection_reward + p_defect * 1

        if ev_defect > ev_coop:
            return 1
        elif ev_coop > ev_defect:
            return 0
        return random.choice(ACTIONS)

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        self.opp_counts[opp_action] += 1

    def reset(self):
        super().reset()
        self.opp_counts = [self.config.fp_smoothing, self.config.fp_smoothing]

    def get_opp_belief(self) -> Dict:
        total = sum(self.opp_counts)
        return {"P(C)": self.opp_counts[0] / total, "P(D)": self.opp_counts[1] / total}


# ─────────────────────────────────────────────
# Independent Q-Learning Agent
# ─────────────────────────────────────────────
class IQLAgent(Agent):
    def __init__(self, agent_id: int, config: Config):
        super().__init__("IQL", agent_id)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.current_state = None
        self.last_action = None

    def select_action(self) -> int:
        state = self.current_state
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            q_vals = self.Q[state]
            max_q = max(q_vals.values())
            best = [a for a, q in q_vals.items() if q == max_q]
            action = random.choice(best)
        self.last_action = action
        return action

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        next_state = (my_action, opp_action)
        state = self.current_state
        max_next_q = max(self.Q[next_state].values())
        td_target = reward + self.config.gamma * max_next_q
        td_error = td_target - self.Q[state][my_action]
        self.Q[state][my_action] += self.config.alpha * td_error
        self.current_state = next_state
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def reset(self):
        super().reset()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.current_state = None
        self.last_action = None


# ─────────────────────────────────────────────
# MiniMax Q-Learning Agent
# ─────────────────────────────────────────────
class MinimaxQAgent(Agent):
    def __init__(self, agent_id: int, config: Config):
        super().__init__("MiniMaxQ", agent_id)
        self.config = config
        self.epsilon = config.epsilon_start
        self.Q: Dict = defaultdict(lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS})
        self.V: Dict = defaultdict(lambda: 0.0)
        self.pi: Dict = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self.current_state = None

    def _solve_minimax(self, state) -> Dict[int, float]:
        Q = self.Q[state]
        q00, q01 = Q[0][0], Q[0][1]
        q10, q11 = Q[1][0], Q[1][1]
        denom = (q00 - q10 - q01 + q11)
        if abs(denom) < 1e-10:
            ev0 = 0.5 * q00 + 0.5 * q01
            ev1 = 0.5 * q10 + 0.5 * q11
            if ev0 > ev1:
                return {0: 1.0, 1: 0.0}
            elif ev1 > ev0:
                return {0: 0.0, 1: 1.0}
            return {0: 0.5, 1: 0.5}
        p = max(0.0, min(1.0, (q11 - q10) / denom))
        return {0: p, 1: 1.0 - p}

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        pi = self.pi[self.current_state]
        return 0 if random.random() < pi[0] else 1

    def update(self, my_action: int, opp_action: int, reward: float, round_num: int):
        super().update(my_action, opp_action, reward, round_num)
        next_state = (my_action, opp_action)
        state = self.current_state
        td_target = reward + self.config.minimax_gamma * self.V[next_state]
        td_error = td_target - self.Q[state][my_action][opp_action]
        self.Q[state][my_action][opp_action] += self.config.minimax_lr * td_error
        self.pi[state] = self._solve_minimax(state)
        pi = self.pi[state]
        self.V[state] = min(
            sum(pi[a] * self.Q[state][a][b] for a in ACTIONS)
            for b in ACTIONS
        )
        self.current_state = next_state
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def reset(self):
        super().reset()
        self.epsilon = self.config.epsilon_start
        self.Q = defaultdict(lambda: {a: {b: 0.0 for b in ACTIONS} for a in ACTIONS})
        self.V = defaultdict(lambda: 0.0)
        self.pi = defaultdict(lambda: {0: 0.5, 1: 0.5})
        self.current_state = None


# ─────────────────────────────────────────────
# Game Runner
# ─────────────────────────────────────────────
class PrisonersDilemmaGame:
    def __init__(self, agent0: Agent, agent1: Agent, config: Config):
        self.agent0 = agent0
        self.agent1 = agent1
        self.config = config
        self.round_log: List[Dict] = []

    def _summary(self) -> Dict:
        n = self.config.n_rounds
        log = self.round_log
        # Final Nash distance (full game empirical)
        p0d = sum(1 for x in log if x["a0"] == 1) / max(len(log), 1)
        p1d = sum(1 for x in log if x["a1"] == 1) / max(len(log), 1)
        nash_dist = compute_nash_distance(p0d, p1d)

        # Tail (last 20%) Nash distance
        tail = log[int(len(log) * 0.8):]
        tp0d = sum(1 for x in tail if x["a0"] == 1) / max(len(tail), 1)
        tp1d = sum(1 for x in tail if x["a1"] == 1) / max(len(tail), 1)
        tail_nash_dist = compute_nash_distance(tp0d, tp1d)

        return {
            "agent0": str(self.agent0),
            "agent1": str(self.agent1),
            "n_rounds": n,
            "total_reward_0": self.agent0.total_reward,
            "total_reward_1": self.agent1.total_reward,
            "avg_reward_0": self.agent0.total_reward / max(n, 1),
            "avg_reward_1": self.agent1.total_reward / max(n, 1),
            "coop_rate_0": self.agent0.cooperation_rate(),
            "coop_rate_1": self.agent1.cooperation_rate(),
            "mutual_coop_rate": self._joint_rate(0, 0),
            "mutual_defect_rate": self._joint_rate(1, 1),
            "exploit_0on1_rate": self._joint_rate(1, 0),
            "exploit_1on0_rate": self._joint_rate(0, 1),
            "nash_distance_overall": nash_dist,
            "nash_distance_tail": tail_nash_dist,
            "nash_distance_p0_defect": p0d,
            "nash_distance_p1_defect": p1d,
        }

    def _joint_rate(self, a0, a1) -> float:
        n = len(self.round_log)
        if n == 0:
            return 0.0
        return sum(1 for log in self.round_log if log["a0"] == a0 and log["a1"] == a1) / n


# ─────────────────────────────────────────────
# Agent Factory
# ─────────────────────────────────────────────
def make_agent(agent_type: str, agent_id: int, config: Config) -> Agent:
    t = agent_type.lower()
    if t == "fp":
        return FictitiousPlayAgent(agent_id, config)
    elif t == "iql":
        return IQLAgent(agent_id, config)
    elif t == "minimax":
        return MinimaxQAgent(agent_id, config)
    raise ValueError(f"Unknown agent type: {agent_type}. Choose: fp, iql, minimax")


# ─────────────────────────────────────────────
# Plotter
# ─────────────────────────────────────────────
def plot_matchup(result: Dict, config: Config, save_path: str):
    """
    4-panel figure for one matchup:
      [0] Cumulative rewards over rounds
      [1] Rolling cooperation rates
      [2] Rolling Nash distance
      [3] Strategy heatmap (empirical joint distribution)
    """
    log = result["round_log"]
    n = len(log)
    rounds = list(range(1, n + 1))
    matchup = result["matchup"]
    colors = MATCHUP_COLORS.get(matchup, ("#1565C0", "#C62828"))
    c0, c1 = colors

    # Pre-compute series
    cum0 = []
    cum1 = []
    s0, s1 = 0.0, 0.0
    coop0_series, coop1_series = [], []

    for entry in log:
        s0 += entry["r0"]; s1 += entry["r1"]
        cum0.append(s0); cum1.append(s1)

    # Rolling cooperation
    window = config.plot_window
    act0 = [x["a0"] for x in log]
    act1 = [x["a1"] for x in log]
    for i in range(n):
        chunk = slice(max(0, i - window + 1), i + 1)
        coop0_series.append(1 - np.mean(act0[chunk]))
        coop1_series.append(1 - np.mean(act1[chunk]))

    # Rolling Nash distance
    d0_series = [1 - c for c in coop0_series]   # defect rate = 1 - coop
    d1_series = [1 - c for c in coop1_series]
    nash_series = [compute_nash_distance(d0_series[i], d1_series[i]) for i in range(n)]

    # Joint action distribution heatmap
    joint = np.zeros((2, 2))
    for entry in log:
        joint[entry["a0"]][entry["a1"]] += 1
    joint /= max(joint.sum(), 1)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    title_kw = dict(fontsize=11, fontweight="bold", color="#212121", pad=10)
    label_kw = dict(fontsize=9, color="#424242")
    tick_kw = dict(labelsize=8)

    # ── Panel 0: Cumulative Rewards ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#F5F5F5")
    ax0.plot(rounds, cum0, color=c0, lw=1.6, label=f"Agent0 ({result['agent0']})")
    ax0.plot(rounds, cum1, color=c1, lw=1.6, ls="--", label=f"Agent1 ({result['agent1']})")
    ax0.set_title("Cumulative Reward", **title_kw)
    ax0.set_xlabel("Round", **label_kw)
    ax0.set_ylabel("Cumulative Reward", **label_kw)
    ax0.tick_params(**tick_kw)
    ax0.legend(fontsize=8, framealpha=0.7)
    ax0.grid(True, alpha=0.3, ls=":")
    # Annotate final values
    ax0.annotate(f"{cum0[-1]:.0f}", xy=(n, cum0[-1]), xytext=(-30, 5),
                 textcoords="offset points", fontsize=8, color=c0)
    ax0.annotate(f"{cum1[-1]:.0f}", xy=(n, cum1[-1]), xytext=(-30, -12),
                 textcoords="offset points", fontsize=8, color=c1)

    # ── Panel 1: Rolling Cooperation Rates ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor("#F5F5F5")
    ax1.plot(rounds, coop0_series, color=c0, lw=1.4, label=f"Agent0 ({result['agent0']})")
    ax1.plot(rounds, coop1_series, color=c1, lw=1.4, ls="--", label=f"Agent1 ({result['agent1']})")
    ax1.axhline(0.0, color="#B71C1C", lw=0.8, ls=":", alpha=0.7, label="Nash (D=1.0)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(f"Rolling Cooperation Rate (window={window})", **title_kw)
    ax1.set_xlabel("Round", **label_kw)
    ax1.set_ylabel("P(Cooperate)", **label_kw)
    ax1.tick_params(**tick_kw)
    ax1.legend(fontsize=8, framealpha=0.7)
    ax1.grid(True, alpha=0.3, ls=":")

    # ── Panel 2: Nash Distance ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#F5F5F5")
    nash_color = "#6A1B9A"
    ax2.plot(rounds, nash_series, color=nash_color, lw=1.4, alpha=0.85)
    ax2.fill_between(rounds, nash_series, alpha=0.15, color=nash_color)
    ax2.axhline(0.0, color="#B71C1C", lw=1.0, ls="--", alpha=0.8, label="Nash EQ (dist=0)")
    # Mark theoretical max distance (both fully cooperate) = sqrt(2) ≈ 1.414
    ax2.axhline(np.sqrt(2), color="#1B5E20", lw=0.8, ls=":", alpha=0.6, label="Max dist (√2)")
    final_nd = nash_series[-1]
    ax2.annotate(f"Final: {final_nd:.3f}", xy=(n, final_nd),
                 xytext=(-60, 8), textcoords="offset points",
                 fontsize=8, color=nash_color,
                 arrowprops=dict(arrowstyle="->", color=nash_color, lw=0.8))
    ax2.set_title(f"Nash Distance (rolling window={window})", **title_kw)
    ax2.set_xlabel("Round", **label_kw)
    ax2.set_ylabel("Distance to Nash EQ", **label_kw)
    ax2.set_ylim(-0.05, np.sqrt(2) + 0.1)
    ax2.tick_params(**tick_kw)
    ax2.legend(fontsize=8, framealpha=0.7)
    ax2.grid(True, alpha=0.3, ls=":")

    # ── Panel 3: Joint Action Heatmap ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(joint, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    labels = ["C (0)", "D (1)"]
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_yticks([0, 1]); ax3.set_yticklabels(labels, fontsize=9)
    ax3.set_xlabel(f"Agent1 ({result['agent1']}) Action", **label_kw)
    ax3.set_ylabel(f"Agent0 ({result['agent0']}) Action", **label_kw)
    ax3.set_title("Empirical Joint Action Distribution", **title_kw)
    for i in range(2):
        for j in range(2):
            val = joint[i, j]
            ax3.text(j, i, f"{val:.2%}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if val > 0.5 else "#212121")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    # Highlight Nash cell (D,D) = [1,1]
    from matplotlib.patches import Rectangle
    ax3.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False,
                             edgecolor="#1B5E20", lw=2.5, label="Nash EQ"))
    ax3.legend(fontsize=8, loc="upper left")

    # Overall figure title
    nd_overall = result["nash_distance_overall"]
    nd_tail = result["nash_distance_tail"]
    fig.suptitle(
        f"Matchup: {matchup}  |  Rounds: {n}  |  "
        f"Nash Dist (overall): {nd_overall:.3f}  |  Nash Dist (tail 20%): {nd_tail:.3f}",
        fontsize=13, fontweight="bold", color="#212121", y=0.98
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Chart saved: {save_path}")


def plot_comparison(results: List[Dict], config: Config, save_path: str):
    """
    Side-by-side comparison chart across all matchups:
      - Average reward bars
      - Final cooperation rate bars
      - Final Nash distance bars
    """
    matchups = [r["matchup"] for r in results]
    n = len(matchups)
    x = np.arange(n)
    bar_w = 0.35

    avg0 = [r["avg_reward_0"] for r in results]
    avg1 = [r["avg_reward_1"] for r in results]
    coop0 = [r["coop_rate_0"] for r in results]
    coop1 = [r["coop_rate_1"] for r in results]
    nash = [r["nash_distance_overall"] for r in results]
    nash_tail = [r["nash_distance_tail"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#FAFAFA")

    palette0 = [MATCHUP_COLORS.get(m, ("#1565C0", "#C62828"))[0] for m in matchups]
    palette1 = [MATCHUP_COLORS.get(m, ("#1565C0", "#C62828"))[1] for m in matchups]

    title_kw = dict(fontsize=11, fontweight="bold", color="#212121", pad=8)
    tick_kw = dict(labelsize=8)

    short_labels = [m.replace(" vs ", "\nvs\n") for m in matchups]

    # ── Average Reward ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    bars0 = ax.bar(x - bar_w / 2, avg0, bar_w, label="Agent0", color=palette0, alpha=0.85, edgecolor="white")
    bars1 = ax.bar(x + bar_w / 2, avg1, bar_w, label="Agent1", color=palette1, alpha=0.85, edgecolor="white")
    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(1.0, color="#B71C1C", lw=1, ls="--", alpha=0.6, label="Nash payoff (1.0)")
    ax.axhline(3.0, color="#1B5E20", lw=1, ls=":", alpha=0.6, label="Coop payoff (3.0)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_title("Average Reward per Round", **title_kw)
    ax.set_ylabel("Avg Reward", fontsize=9)
    ax.tick_params(**tick_kw)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.set_ylim(0, 6.2)

    # ── Cooperation Rates ───────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#F5F5F5")
    bars0 = ax.bar(x - bar_w / 2, [c * 100 for c in coop0], bar_w, label="Agent0",
                   color=palette0, alpha=0.85, edgecolor="white")
    bars1 = ax.bar(x + bar_w / 2, [c * 100 for c in coop1], bar_w, label="Agent1",
                   color=palette1, alpha=0.85, edgecolor="white")
    for bar in list(bars0) + list(bars1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(0.0, color="#B71C1C", lw=1, ls="--", alpha=0.6, label="Nash (0% coop)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_title("Overall Cooperation Rate", **title_kw)
    ax.set_ylabel("Cooperation Rate (%)", fontsize=9)
    ax.tick_params(**tick_kw)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, ls=":")

    # ── Nash Distance ───────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#F5F5F5")
    w = 0.3
    bars_all = ax.bar(x - w / 2, nash, w, label="Overall", color="#7B1FA2", alpha=0.85, edgecolor="white")
    bars_tail = ax.bar(x + w / 2, nash_tail, w, label="Tail (last 20%)", color="#AB47BC", alpha=0.85, edgecolor="white")
    for bar in list(bars_all) + list(bars_tail):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(0.0, color="#B71C1C", lw=1.2, ls="--", alpha=0.8, label="Nash EQ (dist=0)")
    ax.axhline(np.sqrt(2), color="#1B5E20", lw=1, ls=":", alpha=0.6, label="Max dist (√2≈1.41)")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=7.5)
    ax.set_title("Nash Equilibrium Distance", **title_kw)
    ax.set_ylabel("Euclidean Distance to (D,D)", fontsize=9)
    ax.tick_params(**tick_kw)
    ax.set_ylim(0, np.sqrt(2) + 0.15)
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.grid(axis="y", alpha=0.3, ls=":")

    fig.suptitle(
        f"Prisoner's Dilemma — All Matchups Comparison  (rounds={config.n_rounds})",
        fontsize=14, fontweight="bold", color="#212121", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Comparison chart saved: {save_path}")


# ─────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────
class Reporter:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[Dict] = []

    def add(self, matchup_name: str, result: Dict, game: PrisonersDilemmaGame):
        result["matchup"] = matchup_name
        result["round_log"] = game.round_log
        self.results.append(result)

    def print_interval_stats(self, matchup_name: str, result: Dict, interval: int):
        nd = result["nash_distance_overall"]
        print(f"  [{matchup_name}] Round {interval:>6} | "
              f"AvgR0={result['avg_reward_0']:.3f} CoopR0={result['coop_rate_0']:.2%} | "
              f"AvgR1={result['avg_reward_1']:.3f} CoopR1={result['coop_rate_1']:.2%} | "
              f"NashDist={nd:.4f}")

    def print_summary(self):
        sep = "=" * 76
        print(f"\n{sep}")
        print("  PRISONER'S DILEMMA — SIMULATION REPORT")
        print(sep)
        print(f"  Rounds: {self.config.n_rounds} | "
              f"α={self.config.alpha}, γ={self.config.gamma}, "
              f"ε₀={self.config.epsilon_start}→{self.config.epsilon_end} (decay={self.config.epsilon_decay})")
        print(f"  Nash EQ: both Defect (D,D) → payoff (1,1). "
              f"Distance = Euclidean to (p_defect=1, p_defect=1).")
        print(sep)

        for r in self.results:
            nd_o = r["nash_distance_overall"]
            nd_t = r["nash_distance_tail"]
            nd_interpretation = self._interpret_nash(nd_t)
            print(f"\n  ┌─ Matchup: {r['matchup']}")
            print(f"  │  Agent0 ({r['agent0']:>8}) → Avg={r['avg_reward_0']:.3f}, "
                  f"Coop={r['coop_rate_0']:.2%}")
            print(f"  │  Agent1 ({r['agent1']:>8}) → Avg={r['avg_reward_1']:.3f}, "
                  f"Coop={r['coop_rate_1']:.2%}")
            print(f"  │  Mutual Coop={r['mutual_coop_rate']:.2%}  "
                  f"Mutual Defect={r['mutual_defect_rate']:.2%}  "
                  f"Exploit 0→1={r['exploit_0on1_rate']:.2%}  "
                  f"Exploit 1→0={r['exploit_1on0_rate']:.2%}")
            print(f"  │  Nash Distance (overall) : {nd_o:.4f}")
            print(f"  │  Nash Distance (tail 20%): {nd_t:.4f}  → {nd_interpretation}")
            print(f"  └  p_defect: Agent0={r['nash_distance_p0_defect']:.3f}, "
                  f"Agent1={r['nash_distance_p1_defect']:.3f}  "
                  f"(Nash target: 1.000, 1.000)")

        print(f"\n{sep}\n")

    def _interpret_nash(self, nd: float) -> str:
        if nd < 0.05:
            return "Converged to Nash EQ"
        elif nd < 0.3:
            return "Near Nash EQ"
        elif nd < 0.8:
            return "Partial cooperation persists"
        else:
            return "Far from Nash — significant cooperation"

    def convergence_analysis(self):
        print("\n  CONVERGENCE ANALYSIS (last 20% of rounds)")
        print("  " + "-" * 70)
        for r in self.results:
            log = r["round_log"]
            tail = log[int(len(log) * 0.8):]
            if not tail:
                continue
            c0 = sum(1 for x in tail if x["a0"] == 0) / len(tail)
            c1 = sum(1 for x in tail if x["a1"] == 0) / len(tail)
            mc = sum(1 for x in tail if x["a0"] == 0 and x["a1"] == 0) / len(tail)
            md = sum(1 for x in tail if x["a0"] == 1 and x["a1"] == 1) / len(tail)
            nd = r["nash_distance_tail"]
            print(f"  {r['matchup']:22s} | CoopR0={c0:.2%} CoopR1={c1:.2%} "
                  f"| MutualC={mc:.2%} MutualD={md:.2%} | NashDist={nd:.4f}")

    def save_round_log(self, matchup_name: str, filepath: str):
        for r in self.results:
            if r["matchup"] == matchup_name:
                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["round", "a0", "a1", "r0", "r1"])
                    writer.writeheader()
                    writer.writerows(r["round_log"])
                print(f"  → Round log saved to {filepath}")
                return

    def generate_plots(self, plots_dir: str):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"\n  Generating charts → {plots_dir}/")
        for r in self.results:
            safe_name = r["matchup"].replace(" ", "_").lower()
            path = os.path.join(plots_dir, f"{safe_name}.png")
            plot_matchup(r, self.config, path)
        if len(self.results) > 1:
            path = os.path.join(plots_dir, "comparison.png")
            plot_comparison(self.results, self.config, path)


# ─────────────────────────────────────────────
# Simulation Runner
# ─────────────────────────────────────────────
def run_simulation_with_reporting(agent_type_0: str, agent_type_1: str,
                                   config: Config, reporter: Reporter) -> str:
    matchup_name = f"{agent_type_0.upper()} vs {agent_type_1.upper()}"
    print(f"\n  Running: {matchup_name}")

    agent0 = make_agent(agent_type_0, 0, config)
    agent1 = make_agent(agent_type_1, 1, config)
    game = PrisonersDilemmaGame(agent0, agent1, config)
    agent0.reset(); agent1.reset()
    game.round_log = []

    for r in range(1, config.n_rounds + 1):
        a0 = agent0.select_action()
        a1 = agent1.select_action()
        r0, r1 = PAYOFF[(a0, a1)]
        agent0.update(a0, a1, r0, r)
        agent1.update(a1, a0, r1, r)
        game.round_log.append({"round": r, "a0": a0, "a1": a1, "r0": r0, "r1": r1})

        if r % config.report_interval == 0 or r == config.n_rounds:
            interim = game._summary()
            reporter.print_interval_stats(matchup_name, interim, r)

    result = game._summary()
    reporter.add(matchup_name, result, game)
    return matchup_name


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Repeated Prisoner's Dilemma — FP & RL with Nash tracking & charts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--matchups", nargs="+",
                        choices=["fp_vs_fp", "iql_vs_iql", "minimax_vs_minimax",
                                 "iql_vs_fp", "minimax_vs_fp", "iql_vs_minimax", "all"],
                        default=["all"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--fp-smoothing", type=float, default=1.0)
    parser.add_argument("--minimax-lr", type=float, default=0.1)
    parser.add_argument("--minimax-gamma", type=float, default=0.9)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--plot-window", type=int, default=50,
                        help="Rolling window size for charts")    
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")    
    parser.add_argument("--save-round-logs", action="store_true")
    parser.add_argument("--convergence", action="store_true")    

    args = parser.parse_args()

    config = Config(
        n_rounds=args.rounds,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        fp_smoothing=args.fp_smoothing,
        minimax_lr=args.minimax_lr,
        minimax_gamma=args.minimax_gamma,
        report_interval=args.report_interval,
        plot_window=args.plot_window
    )

    ALL_MATCHUPS = {
        "fp_vs_fp":             ("fp",      "fp"),
        "iql_vs_iql":           ("iql",     "iql"),
        "minimax_vs_minimax":   ("minimax", "minimax"),
        "iql_vs_fp":            ("iql",     "fp"),
        "minimax_vs_fp":        ("minimax", "fp"),
        "iql_vs_minimax":       ("iql",     "minimax"),
    }
    selected = list(ALL_MATCHUPS.keys()) if "all" in args.matchups else args.matchups

    reporter = Reporter(config)

    print("\n" + "=" * 76)
    print("  PRISONER'S DILEMMA SIMULATION")
    print("  Payoff: C,C=(3,3) | C,D=(0,5) | D,C=(5,0) | D,D=(1,1)")
    print("  Nash EQ (stage game): (D,D) — distance measures deviation from it")
    print("=" * 76)

    for key in selected:
        t0, t1 = ALL_MATCHUPS[key]
        name = run_simulation_with_reporting(t0, t1, config, reporter)
        if args.save_round_logs:
            reporter.save_round_log(name, f"round_log_{key}.csv")

    reporter.print_summary()

    if args.convergence:
        reporter.convergence_analysis()

    if not args.no_plots:
        reporter.generate_plots(os.getcwd())

if __name__ == "__main__":
    main()
